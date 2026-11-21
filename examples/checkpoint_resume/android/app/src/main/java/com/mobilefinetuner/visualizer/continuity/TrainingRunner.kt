package com.mobilefinetuner.visualizer.continuity

import android.app.ActivityManager
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.BatteryManager
import android.os.PowerManager
import org.json.JSONObject
import java.io.File
import java.io.RandomAccessFile
import java.util.concurrent.TimeUnit

data class Resources(val charging: Boolean, val battery: Int, val thermal: Int,
                     val lowMemory: Boolean, val availableMb: Long, val diskMb: Long,
                     val totalMb: Long = 0, val lowThresholdMb: Long = 0) {
    fun nonMemoryBlock(requireCharging: Boolean): String? = when {
        requireCharging && !charging -> "waiting for charging"
        battery < 20 -> "battery below 20%"
        thermal >= PowerManager.THERMAL_STATUS_SEVERE -> "thermal pressure"
        diskMb < 128 -> "storage pressure"
        else -> null
    }
    companion object {
        fun read(c: Context): Resources {
            val battery = c.registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
            val level = battery?.getIntExtra(BatteryManager.EXTRA_LEVEL, -1) ?: -1
            val scale = battery?.getIntExtra(BatteryManager.EXTRA_SCALE, 100) ?: 100
            val memory = ActivityManager.MemoryInfo()
            c.getSystemService(ActivityManager::class.java).getMemoryInfo(memory)
            return Resources((battery?.getIntExtra(BatteryManager.EXTRA_PLUGGED, 0) ?: 0) != 0,
                if (scale > 0) level * 100 / scale else -1,
                c.getSystemService(PowerManager::class.java).currentThermalStatus,
                memory.lowMemory, memory.availMem / 1048576, c.filesDir.usableSpace / 1048576,
                memory.totalMem / 1048576, memory.threshold / 1048576)
        }
    }
}

class TrainingRunner(private val context: Context) {
    companion object {
        // ops::resume::CheckpointsExhausted::exit_code (all recorded candidates failed validation).
        private const val CHECKPOINTS_EXHAUSTED = 65
        fun executionBudgetMs(slice: Int, pauseGrace: Int): Long {
            require(slice in 1..120) { "Slice must be 1..120 seconds" }
            require(pauseGrace in 30..420) { "Pause grace must be 30..420 seconds" }
            return (slice + pauseGrace) * 1000L
        }
        fun wakeBudgetMs(slice: Int, pauseGrace: Int): Long = executionBudgetMs(slice, pauseGrace) + 10000L
        fun exhaustedWithoutProgress(before: Int, after: Int, elapsedMs: Long, slice: Int,
                                     stoppedByControl: Boolean): Boolean =
            after == before && elapsedMs >= slice * 1000L && !stoppedByControl
        internal fun sessionStartUpdate(run: File, session: String): Int {
            require(session.matches(Regex("[a-zA-Z0-9_-]+"))) { "Invalid committing session" }
            return File(run, "sessions/$session/events.jsonl").bufferedReader().use { reader ->
                val start = JSONObject(reader.readLine())
                require(start.getString("session_id") == session && start.getString("type") == "session_start")
                if (!start.getBoolean("restoring")) 0 else {
                    val resume = JSONObject(reader.readLine())
                    require(resume.getString("session_id") == session && resume.getString("type") == "resume")
                    resume.getInt("completed_updates").also { require(it >= 0) }
                }
            }
        }
        fun errorTail(file: File): String = runCatching {
            RandomAccessFile(file, "r").use { log ->
                val size = minOf(log.length(), 8192L).toInt()
                log.seek(log.length() - size)
                val bytes = ByteArray(size)
                log.readFully(bytes)
                val lines = bytes.toString(Charsets.UTF_8).lineSequence().filter { it.isNotBlank() }.toList()
                val errors = lines.filter { it.contains(Regex("error|exception|fatal|invalid|failed|cannot|missing", RegexOption.IGNORE_CASE)) }
                (if (errors.isNotEmpty()) errors else lines).takeLast(8).joinToString("\n").takeLast(1200)
            }
        }.getOrDefault("")
    }
    enum class Outcome { COMPLETED, SUSPENDED, WAITING, DISABLED, BUSY, FAILED }
    fun run(id: String, background: Boolean, cancelled: () -> Boolean = { false },
            stopAfter: Int = 0, expectedGeneration: Long? = null, rejectBusy: Boolean = true): Outcome {
        val store = TaskStore(context)
        val dir = store.task(id)
        val generation = expectedGeneration ?: store.control(id).optLong("resume_generation")
        try {
            return store.withTrainingExecution(id, onBusy = { sameTask ->
                if (rejectBusy && !background && !sameTask) store.rejectStart(id, generation)
                Outcome.BUSY
            }) withActiveExecution@{
                store.respectSystemStop(id)
                val control = store.control(id)
                if (control.optLong("resume_generation") != generation || cancelled() ||
                    !control.optBoolean("enabled") || (background && !control.optBoolean("automatic")))
                    return@withActiveExecution Outcome.DISABLED
                if (store.status(id).optString("committed_status") == "completed") return@withActiveExecution Outcome.COMPLETED
                val spec = store.spec(id)
                TaskStore.validateSpec(spec)
                val guard = MemoryGuard(spec.optInt("min_available_mb", 512))
                val resources = Resources.read(context)
                val memoryBlock = guard.beforeStart(resources)
                if (memoryBlock != null) {
                    store.holdForMemory(id, generation, memoryBlock)
                    return@withActiveExecution Outcome.WAITING
                }
                val reason = resources.nonMemoryBlock(control.optBoolean("charging", true))
                if (reason != null) { store.note(id, "waiting", reason, generation); return@withActiveExecution Outcome.WAITING }
                val run = File(dir, "run")
                val binary = File(context.applicationInfo.nativeLibraryDir, "libmft_${spec.getString("model")}.so")
                val command = mutableListOf(binary.absolutePath)
                val latest = File(run, "LATEST")
                val resumePointer = if (latest.isFile) latest.readText() else null
                if (resumePointer != null) {
                    command += listOf("--resume_state", run.absolutePath)
                    if (control.has("resume_microbatch")) {
                        val size = control.get("resume_microbatch")
                        require(size is Int) { "Invalid persisted physical batch" }
                        TaskStore.validateMicrobatch(spec.getInt("batch_size"), size)
                        command += listOf("--resume_microbatch", size.toString())
                    }
                } else {
                    require(!control.has("resume_microbatch")) { "Physical batch override requires a checkpoint" }
                    require(!store.hasAttemptArtifacts(id)) { "No committed checkpoint after an earlier attempt: Retry as new task; partial files retained" }
                    command += listOf("--model_dir", File(dir, "model").absolutePath,
                        "--jsonl_train", File(dir, "train.jsonl").absolutePath,
                        "--jsonl_valid", File(dir, "valid.jsonl").absolutePath, "--run_dir", run.absolutePath,
                        "--lora_dropout", "0", "--lr_scheduler", "cosine", "--warmup_steps", "2",
                        "--log_interval", "1", "--eval_interval", "3", "--eval_batches", "1",
                        "--save_every", "3", "--seed", "42", "--shuffle", "--save_initial_adapter")
                    for (key in listOf("steps", "batch_size", "seq_len", "lora_r", "grad_accum"))
                        command += listOf("--$key", spec.getInt(key).toString())
                    command += listOf("--lr", spec.getDouble("lr").toString())
                    command += listOf("--lora_alpha", (spec.getInt("lora_r") * 2).toString())
                    if (File(dir, "initial.safetensors").isFile)
                        command += listOf("--resume_from", File(dir, "initial.safetensors").absolutePath)
                    if (spec.optBoolean("activation_checkpointing")) command += "--activation_checkpointing"
                    if (spec.optBoolean("memory_efficient_attention")) command += "--memory_efficient_attention"
                    if (spec.optBoolean("sharding")) command += listOf("--shard_enable", "--shard_budget_mb",
                        spec.optInt("shard_budget_mb", 768).toString(), "--shard_fp16_disk", "false")
                }
                val slice = control.optInt("slice_seconds", spec.optInt("slice_seconds", 30))
                val pauseGrace = control.optInt("pause_grace_seconds", spec.optInt("pause_grace_seconds", 30))
                val executionBudget = executionBudgetMs(slice, pauseGrace)
                command += listOf("--checkpoint_every_updates", "1", "--slice_seconds", slice.toString(),
                    "--suspend_after_updates", stopAfter.toString())
                val wake = context.getSystemService(PowerManager::class.java)
                    .newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "mft:training-slice")
                var child: NativeOwnedProcess? = null
                var attemptLog: File? = null
                val started = android.os.SystemClock.elapsedRealtime()
                try {
                    // The legacy native config reader tolerates malformed JSON via defaults.
                    // Reject broken input before allocating a model or promising recovery.
                    JSONObject(File(dir, "model/config.json").readText())
                    wake.acquire(wakeBudgetMs(slice, pauseGrace))
                    val attempt = File(dir, "attempts")
                    val log = File(attempt, "${System.currentTimeMillis()}.log")
                    attemptLog = log
                    val spawn = NativeOwnedProcess.prepare(command, dir, log, spec.optInt("threads", 4))
                    val cpu = if (TaskStore.cpuContentionEnabled(control, spec)) CpuContentionGuard() else null
                    val scheduler = cpu?.let { ChildScheduler(run, spec.optInt("threads", 4)) }
                    child = store.launchIfCurrent(id, generation, background, cancelled) {
                        check(attempt.isDirectory || attempt.mkdirs()) { "Cannot prepare attempt log" }
                        spawn()
                    } ?: return@withActiveExecution Outcome.DISABLED
                    store.note(id, "running", "native child active; pid is recorded in session events", generation)
                    val deadline = started + executionBudget
                    var forced = false
                    var stoppedByControl = false
                    var notedReason: String? = null
                    while (!child.waitFor(200, TimeUnit.MILLISECONDS)) {
                        val ctl = store.control(id)
                        // Kill only the child owned by this slice, never a PID from an old run.
                        if (ctl.optLong("release_through", -1L) >= generation ||
                            ctl.optLong("current_focus_epoch") != control.optLong("focus_epoch")) {
                            child.terminate()
                            store.note(id, "released", "Released resources; only committed updates are retained.", generation)
                            return@withActiveExecution Outcome.DISABLED
                        }
                        val sample = Resources.read(context)
                        val now = android.os.SystemClock.elapsedRealtime()
                        val pressure = guard.duringTraining(sample, now)
                        if (pressure != null && ctl.optBoolean("enabled"))
                            store.holdForMemory(id, generation, pressure)
                        val blocked = pressure ?: sample.nonMemoryBlock(ctl.optBoolean("charging", true))
                        val stop = cancelled() || !ctl.optBoolean("enabled") || ctl.optLong("resume_generation") != generation
                        val cpuReason = cpu?.poll(now, !stop && !stoppedByControl && blocked == null &&
                            guard.beforeStart(sample) == null) { scheduler?.read() }
                        if (cpuReason != null) store.holdForCpu(id, generation, cpuReason)
                        if (stop || blocked != null || cpuReason != null) {
                            stoppedByControl = true
                            store.requestSuspend(id)
                            val reasonText = blocked ?: cpuReason ?: "pause requested"
                            if (notedReason != reasonText && store.note(id, "suspending", reasonText, generation))
                                notedReason = reasonText
                        }
                        if (android.os.SystemClock.elapsedRealtime() > deadline) {
                            forced = true
                            child.terminate()
                            break
                        }
                    }
                    val exit = child.exitValue()
                    if (forced || exit != 0) {
                        store.fail(id, "exit=$exit, deadline=$forced\n${errorTail(log)}\nFull log: attempts/${log.name}",
                            generation, resumePointer.takeIf { !forced && exit == CHECKPOINTS_EXHAUSTED })
                        return@withActiveExecution Outcome.FAILED
                    }
                    val snapshot = store.status(id)
                    val status = snapshot.optString("committed_status")
                    require(status in listOf("completed", "suspended")) { "Native process exited without a completed or suspended checkpoint" }
                    val manual = cancelled() || !store.control(id).optBoolean("enabled")
                    if (status == "suspended" && exhaustedWithoutProgress(
                            sessionStartUpdate(run, snapshot.getString("committing_session")), snapshot.getInt("committed_updates"),
                            android.os.SystemClock.elapsedRealtime() - started, slice, stoppedByControl || manual)) {
                        store.fail(id, "No update: slice shorter than model loading/restoration. Increase Slice seconds and Resume.", generation)
                        return@withActiveExecution Outcome.FAILED
                    }
                    store.note(id, status, expectedGeneration = generation)
                    if (status == "completed") Outcome.COMPLETED else Outcome.SUSPENDED
                } catch (e: Exception) {
                    store.fail(id, "${e.message ?: e.javaClass.simpleName}\n${attemptLog?.let { errorTail(it) } ?: ""}", generation)
                    Outcome.FAILED
                } finally {
                    try { child?.terminate() } finally { if (wake.isHeld) wake.release() }
                }
            }
        } catch (e: Exception) {
            store.fail(id, e.message ?: e.javaClass.simpleName, generation)
            return Outcome.FAILED
        }
    }
}
