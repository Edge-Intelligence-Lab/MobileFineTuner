package com.mobilefinetuner.visualizer.continuity

import android.app.ActivityManager
import android.app.ApplicationExitInfo
import android.content.Context
import android.net.Uri
import android.os.Build
import android.util.AtomicFile
import androidx.documentfile.provider.DocumentFile
import org.json.JSONObject
import java.io.File
import java.io.RandomAccessFile
import java.util.UUID
import java.util.zip.ZipEntry
import java.util.zip.ZipOutputStream

class TaskStore(val context: Context) {
    val root = File(context.filesDir, "training").apply { mkdirs() }
    fun task(id: String): File {
        require(id.matches(Regex("[a-zA-Z0-9_-]{1,80}"))) { "Invalid task ID" }
        return File(root, id)
    }
    fun list(): List<String> = root.listFiles()?.filter { File(it, "spec.json").isFile }
        ?.map { it.name }?.sorted() ?: emptyList()
    fun spec(id: String) = JSONObject(File(task(id), "spec.json").readText())
    // Activity-only preference, never consulted by the worker or native training.
    fun lastSelected(ids: List<String>): String? =
        context.getSharedPreferences("training_ui", Context.MODE_PRIVATE)
            .getString("selected_task", null)?.takeIf { it in ids } ?: ids.lastOrNull()
    fun rememberSelected(id: String) {
        context.getSharedPreferences("training_ui", Context.MODE_PRIVATE)
            .edit().putString("selected_task", id).apply()
    }
    fun control(id: String): JSONObject = locked(id) {
        val value = readControl(id)
        FocusWindow(context).withGate { focus ->
            if (!focus.permits(value)) value.put("enabled", false).put("automatic", false)
            value.put("current_focus_epoch", focus.epoch)
        }
    }
    fun canRetryInBackground(id: String, generation: Long): Boolean {
        val ctl = control(id)
        return ctl.optBoolean("enabled") && ctl.optBoolean("automatic") &&
            ctl.optLong("resume_generation") == generation
    }
    private fun readControl(id: String) = try {
        JSONObject(AtomicFile(File(task(id), "control.json")).openRead().bufferedReader().use { it.readText() })
    } catch (_: java.io.FileNotFoundException) {
        JSONObject().put("enabled", false).put("automatic", false).put("charging", true)
            .put("explicit_resume_ms", 0L)
    }
    private fun <T> locked(id: String, block: () -> T): T {
        task(id).mkdirs()
        // Serialize both threads and the foreground service's separate process.
        synchronized(mutex) {
            RandomAccessFile(File(task(id), "control.lock"), "rw").use { file ->
                file.channel.lock().use { return block() }
            }
        }
    }
    fun updateControl(id: String, edit: (JSONObject) -> Unit) = locked(id) {
        val value = readControl(id)
        edit(value)
        write(File(task(id), "control.json"), value)
    }
    fun setCpuContentionGuard(id: String, enabled: Boolean) = updateControl(id) {
        it.put("cpu_contention_guard", enabled)
    }
    private fun bootNumber() = android.provider.Settings.Global.getInt(
        context.contentResolver, android.provider.Settings.Global.BOOT_COUNT)
    fun allow(id: String, automatic: Boolean, charging: Boolean): Long =
        checkNotNull(admitStart(id, automatic, charging))
    fun admitStart(id: String, automatic: Boolean, charging: Boolean, issued: Long? = null,
                   slice: Int? = null, grace: Int? = null, resumeMicrobatch: Int? = null): Long? = locked(id) {
        val value = readControl(id)
        val boot = bootNumber()
        val focus = FocusWindow(context).withGate { it }
        require(focus.remainingMs == 0L) { "Focus hour active. End focus before starting training." }
        if (issued != null && focus.sameBoot && issued <= focus.resumeAfterNs) return@locked null
        value.put("focus_epoch", focus.epoch)
        if (issued != null && value.optInt("intent_boot", -1) == boot &&
            value.optLong("intent_elapsed_ns", -1) >= issued) return@locked null
        if (resumeMicrobatch != null) {
            validateMicrobatch(spec(id).getInt("batch_size"), resumeMicrobatch)
            val committed = checkpoint(id)
            require(committed != null && committed.getString("committed_status") != "completed") {
                "A physical batch can only be selected for an unfinished checkpoint"
            }
            value.put("resume_microbatch", resumeMicrobatch)
        }
        value.put("intent_boot", boot).put("intent_elapsed_ns", issued ?: android.os.SystemClock.elapsedRealtimeNanos())
        slice?.let { value.put("slice_seconds", it) }
        grace?.let { value.put("pause_grace_seconds", it) }
        val generation = value.optLong("resume_generation") + 1L
        value.remove("exhausted_pointer")
        value.remove("memory_hold_reason")
        value.remove("cpu_hold_reason")
        value.put("enabled", true).put("automatic", automatic).put("charging", charging)
            .put("resume_generation", generation)
            .put("explicit_resume_ms", System.currentTimeMillis())
        write(File(task(id), "control.json"), value)
        generation
    }
    fun pause(id: String, immediately: Boolean = false): Long = locked(id) {
        pauseLocked(id, readControl(id), immediately)
    }
    internal fun pauseIfCurrent(id: String, generation: Long): Boolean = locked(id) {
        val ctl = readControl(id)
        if (generation != ctl.optLong("resume_generation")) return@locked false
        pauseLocked(id, ctl, false)
        true
    }
    private fun pauseLocked(id: String, ctl: JSONObject, immediately: Boolean): Long {
        val generation = ctl.optLong("resume_generation")
        ctl.remove("memory_hold_reason")
        ctl.remove("cpu_hold_reason")
        if (immediately) ctl.put("release_through", generation)
        disableLocked(id, ctl)
        return generation
    }
    internal fun dispatchStartIfCurrent(id: String, generation: Long, dispatch: () -> Unit): Boolean = locked(id) {
        val ctl = readControl(id)
        FocusWindow(context).withGate { focus ->
            if (!ctl.optBoolean("enabled") || ctl.optLong("resume_generation") != generation || !focus.permits(ctl))
                false
            else {
                if (!isExecuting(id)) writeNote(id, "starting", "Start requested; awaiting execution admission", generation)
                dispatch()
                true
            }
        }
    }
    fun holdForMemory(id: String, generation: Long, reason: String) =
        holdForResource(id, generation, "memory_hold_reason", reason)
    fun holdForCpu(id: String, generation: Long, reason: String) =
        holdForResource(id, generation, "cpu_hold_reason", reason)
    private fun holdForResource(id: String, generation: Long, key: String, reason: String): Boolean = locked(id) {
        val ctl = readControl(id)
        if (!ctl.optBoolean("enabled") || ctl.optLong("resume_generation") != generation) return@locked false
        // Persist the hold before signalling the child: a worker/reboot must not restart it.
        disableLocked(id, ctl.put(key, reason), userIntent = false)
        true
    }
    private fun disableLocked(id: String, value: JSONObject, userIntent: Boolean = true) {
        if (userIntent)
            value.put("intent_boot", bootNumber()).put("intent_elapsed_ns", android.os.SystemClock.elapsedRealtimeNanos())
        value.put("enabled", false).put("automatic", false)
        write(File(task(id), "control.json"), value)
        requestSuspend(id)
    }
    fun requestSuspend(id: String) {
        val run = File(task(id), "run")
        // Idempotent during a long update. The managed launcher retires old markers.
        if (run.isDirectory) File(run, "SUSPEND").createNewFile()
    }
    private fun writeNote(id: String, state: String, detail: String, generation: Long) =
        write(File(task(id), "ui_state.json"), JSONObject().put("state", state).put("detail", detail.take(2400))
            .put("resume_generation", generation).put("time_ms", System.currentTimeMillis()))
    fun note(id: String, state: String, detail: String = "", expectedGeneration: Long? = null): Boolean = locked(id) {
        val generation = readControl(id).optLong("resume_generation")
        if (expectedGeneration != null && generation != expectedGeneration) return@locked false
        writeNote(id, state, detail, generation)
        true
    }
    internal fun <T : Any> launchIfCurrent(id: String, generation: Long, background: Boolean,
                                            cancelled: () -> Boolean, start: () -> T): T? = locked(id) {
        FocusWindow(context).withGate { focus ->
            val ctl = readControl(id)
            if (ctl.optLong("resume_generation") != generation || !ctl.optBoolean("enabled") ||
                (background && !ctl.optBoolean("automatic")) || !focus.permits(ctl) || cancelled()) null
            else {
                val run = File(task(id), "run")
                check(run.isDirectory || run.mkdirs()) { "Cannot prepare run directory" }
                val marker = File(run, "SUSPEND")
                check(!marker.exists() || marker.delete()) { "Cannot retire stale suspend marker" }
                start()
            }
        }
    }

    // Only lease transitions and short status probes enter here, never training or task control.
    // Otherwise a reader's successful tryLock can look exactly like a competing trainer.
    private fun <T> leaseGate(block: () -> T): T = synchronized(leaseMutex) {
        RandomAccessFile(File(root, "lease-transition.lock"), "rw").use { file ->
            file.channel.lock().use { block() }
        }
    }
    private val executionLock get() = File(root, "execution.lock")
    private fun activeLock(id: String) = File(task(id), "active.lock")

    // Caller holds leaseGate for acquire and close. Never open a second descriptor to
    // our own POSIX record lock: closing it could silently unlock the first descriptor.
    private fun acquire(path: File): java.io.Closeable? {
        val key = path.absolutePath
        if (key in localLeases) return null
        val file = RandomAccessFile(path, "rw")
        try {
            if (file.channel.tryLock() == null) { file.close(); return null }
            localLeases.add(key)
            return java.io.Closeable {
                try { file.close() } finally { localLeases.remove(key) }
            }
        } catch (e: Throwable) { file.close(); throw e }
    }
    private fun busy(path: File): Boolean = acquire(path)?.use { false } ?: true
    private fun <T : Any> withLease(path: File, block: () -> T): T? {
        val lease = leaseGate { acquire(path) } ?: return null
        try { return block() } finally { leaseGate { lease.close() } }
    }
    fun <T : Any> withExecutionLease(block: () -> T): T? = withLease(executionLock, block)
    fun executionBusy(): Boolean = leaseGate { busy(executionLock) }
    // Keep the status read inside the short transition, including closing the probe.
    internal fun <T> readActiveExecution(id: String, read: (Boolean) -> T): T = leaseGate {
        val probe = acquire(activeLock(id))
        try { read(probe == null) } finally { probe?.close() }
    }
    fun isExecuting(id: String): Boolean = readActiveExecution(id) { it }
    fun anotherTaskExecuting(id: String): Boolean = leaseGate { busy(executionLock) && !busy(activeLock(id)) }
    internal fun <T : Any> withTrainingExecution(id: String, onBusy: (Boolean) -> T, block: () -> T): T {
        val path = activeLock(id)
        var sameTask = false
        val leases = leaseGate {
            val global = acquire(executionLock)
            if (global == null) {
                sameTask = busy(path)
                null
            } else {
                try {
                    val active = acquire(path)
                    if (active != null) global to active else {
                        sameTask = true
                        global.close()
                        null
                    }
                } catch (e: Throwable) { global.close(); throw e }
            }
        } ?: return onBusy(sameTask)
        try { return block() } finally {
            leaseGate { try { leases.second.close() } finally { leases.first.close() } }
        }
    }
    fun <T : Any> withActiveExecution(id: String, block: () -> T): T =
        checkNotNull(withLease(File(task(id), "active.lock"), block)) { "Task already has an active execution" }
    fun rejectStart(id: String, expectedGeneration: Long? = null) = locked(id) {
        val value = readControl(id)
        if (expectedGeneration != null && value.optLong("resume_generation") != expectedGeneration) return@locked
        // The competing owner may already have exited and this task won a new lease.
        if (!value.optBoolean("enabled") || isExecuting(id)) return@locked
        value.put("enabled", false).put("automatic", false)
        write(File(task(id), "control.json"), value)
        requestSuspend(id)
        writeNote(id, "rejected", "Start rejected: another task is active. Pause it, then press Start / Resume again. This request was not queued.",
            value.optLong("resume_generation"))
    }
    fun hasAttemptArtifacts(id: String): Boolean =
        File(task(id), "run").listFiles()?.isNotEmpty() == true ||
            File(task(id), "attempts").listFiles()?.isNotEmpty() == true

    // This is a cheap UI metadata check, not a substitute for native tensor/asset validation.
    fun checkpoint(id: String): JSONObject? {
        val run = File(task(id), "run")
        val latest = runCatching { JSONObject(File(run, "LATEST").readText()).also {
            require(it.getInt("schema") == 1)
        } }.getOrNull() ?: return null
        for (key in listOf("current", "previous")) {
            val candidate = runCatching {
                val generation = latest.getString(key)
                require(generation.matches(Regex("[a-zA-Z0-9_-]+")))
                val dir = File(run, "training_state/$generation")
                val manifest = JSONObject(File(dir, "manifest.json").readText())
                require(manifest.getInt("schema") == 1 && manifest.getString("generation") == generation)
                for (name in listOf("state.json", "state.safetensors")) {
                    val file = File(dir, name)
                    require(file.isFile && file.length() > 0 && file.length() == manifest.getJSONObject("files").getLong(name))
                }
                val state = JSONObject(File(dir, "state.json").readText())
                val trainer = state.getJSONObject("trainer")
                val updates = trainer.getInt("completed_updates")
                require(state.getInt("schema") == 1 && updates >= 0 && updates <= trainer.getInt("total_steps"))
                require(updates == manifest.getInt("completed_updates"))
                require(state.getString("status") in listOf("running", "suspended", "completed"))
                JSONObject().put("committed_updates", updates).put("committed_status", state.getString("status"))
                    .put("committed_physical_batch", trainer.opt("physical_batch"))
                    .put("checkpoint_generation", generation).put("checkpoint_fallback", key == "previous")
                    .put("committing_session", manifest.optString("session_id"))
            }.getOrNull()
            if (candidate != null) return candidate
        }
        return null
    }
    private fun noCommittedPointer(id: String): Boolean = try {
        android.system.Os.stat(File(task(id), "run/LATEST").path)
        false
    } catch (e: android.system.ErrnoException) {
        e.errno == android.system.OsConstants.ENOENT
    }
    fun fail(id: String, reason: String, expectedGeneration: Long? = null,
             exhaustedPointer: String? = null, onlyIfEnabled: Boolean = false): Boolean = locked(id) {
        val value = readControl(id)
        if (onlyIfEnabled && (!value.optBoolean("enabled") ||
                !FocusWindow(context).withGate { it.permits(value) })) return@locked false
        if (expectedGeneration != null && value.optLong("resume_generation") != expectedGeneration) {
            write(File(task(id), "last_stale_failure.json"), JSONObject()
                .put("resume_generation", expectedGeneration).put("detail", reason.take(1800))
                .put("time_ms", System.currentTimeMillis()))
            return@locked false
        }
        value.put("enabled", false).put("automatic", false).put("exhausted_pointer", exhaustedPointer)
        write(File(task(id), "control.json"), value)
        requestSuspend(id)
        val recovery = if (exhaustedPointer != null)
            "Native validation rejected both committed checkpoints. Retry as new task keeps the original files."
        else if (!noCommittedPointer(id))
            "Checkpoint candidate retained; Start / Resume validates it natively."
        else if (!hasAttemptArtifacts(id)) "No native attempt was made. Start / Resume retries this same task after fixing the error."
        else "No readable committed checkpoint. Retry as new task reinitializes training; original failure files are retained."
        writeNote(id, "failed", "${reason.take(1800)}\n$recovery", value.optLong("resume_generation"))
        true
    }
    fun status(id: String): JSONObject {
        val result = runCatching { JSONObject(File(task(id), "ui_state.json").readText()) }
            .getOrElse { JSONObject().put("state", "ready") }
        val history = result.optString("state")
        result.put("historical_state", history).put("historical_detail", result.optString("detail"))
        val committed = checkpoint(id)
        val attempted = hasAttemptArtifacts(id)
        committed?.keys()?.forEach { result.put(it, committed.get(it)) }
        val ctl = control(id)
        val exhausted = ctl.has("exhausted_pointer") && runCatching {
            File(task(id), "run/LATEST").readText() == ctl.getString("exhausted_pointer")
        }.getOrDefault(false)
        val needsInitialization = attempted && noCommittedPointer(id)
        val active = isExecuting(id)
        val enabled = ctl.optBoolean("enabled")
        result.put("control", ctl).put("execution_active", active).put("checkpoint_available", committed != null && !exhausted)
        val state = when {
            active -> if (!enabled || history == "suspending") "suspending" else "running"
            history in listOf("failed", "rejected") -> history
            committed?.optString("committed_status") == "completed" -> "completed"
            !enabled && ctl.has("memory_hold_reason") -> "memory_hold"
            !enabled && ctl.has("cpu_hold_reason") -> "cpu_hold"
            !enabled -> if (history == "suspended" && committed?.optString("committed_status") == "suspended") "suspended"
                else if (history == "ready") "ready" else "paused"
            ctl.optBoolean("automatic") -> "waiting"
            history in listOf("running", "suspending", "starting", "handoff") -> "interrupted"
            else -> history
        }
        result.put("state", state)
        when {
            state == "cpu_hold" -> result.put("detail", ctl.getString("cpu_hold_reason") +
                if (needsInitialization || exhausted) " No usable checkpoint; use Retry as new task."
                else " Training stopped. Press Start / Resume when ready.")
            state == "suspending" && ctl.has("cpu_hold_reason") -> result.put("detail",
                ctl.getString("cpu_hold_reason") + " Waiting for the trainer to checkpoint and exit.")
            state == "memory_hold" -> result.put("detail", ctl.getString("memory_hold_reason") +
                if (needsInitialization) " No readable checkpoint; use Retry as new task."
                else " Training is not running. Press Start / Resume when ready.")
            state == "suspending" && ctl.has("memory_hold_reason") -> result.put("detail",
                ctl.getString("memory_hold_reason") + " Waiting for the trainer to checkpoint and exit.")
            state == "interrupted" -> result.put("detail", if (committed != null)
                "Execution stopped; Start / Resume validates the retained checkpoint."
                else if (needsInitialization) "Execution stopped before a readable checkpoint; Retry as new task is required."
                else if (attempted) "Start / Resume retries validation of the recorded checkpoint."
                else "Start was interrupted before a native attempt; Start / Resume retries this same task.")
            state == "paused" -> result.put("detail", "Continuation disabled; no live execution. No new checkpoint is implied.")
            state == "waiting" -> result.put("detail", if (ctl.optBoolean("automatic"))
                "No live execution. Background continuation enabled; Android may delay work. ${if (history == "waiting") result.optString("historical_detail") else ""}"
                else "${result.optString("historical_detail")} No automatic retry; press Start / Resume after resolving the condition.")
            state == "suspending" && !enabled -> result.put("detail", "Pause requested; waiting for native exit and checkpoint result.")
        }
        val needsNew = !active && (exhausted || needsInitialization)
        result.put("retry_as_new", needsNew)
        return result
    }
    fun retryAsNew(id: String): String {
        return checkNotNull(withExecutionLease {
                check(status(id).optBoolean("retry_as_new")) { "Use Start / Resume for a task with a checkpoint" }
                val original = spec(id)
                validateSpec(original)
                val next = UUID.randomUUID().toString()
                val dir = task(next).apply { mkdirs() }
                try {
                    val names = listOf("model/config.json", "model/model.safetensors", "train.jsonl", "valid.jsonl") +
                        if (File(task(id), "initial.safetensors").isFile) listOf("initial.safetensors") else emptyList()
                    for (name in names) {
                        val source = File(task(id), name)
                        require(source.isFile) { "Missing retry asset: $name; re-import the experiment folder" }
                        require(dir.usableSpace > source.length() + 128L * 1024 * 1024) { "Insufficient storage for $name" }
                        val target = File(dir, name).apply { parentFile!!.mkdirs() }
                        source.inputStream().use { input -> target.outputStream().use { output ->
                            input.copyTo(output, 1024 * 1024); output.fd.sync()
                        } }
                    }
                    File(task(id), "spec.json").copyTo(File(dir, "spec.json"))
                    val previousControl = control(id)
                    updateControl(next) { ctl ->
                        ctl.put("enabled", false).put("automatic", false)
                            .put("charging", previousControl.optBoolean("charging", true))
                        if (previousControl.has("slice_seconds")) ctl.put("slice_seconds", previousControl.getInt("slice_seconds"))
                        if (previousControl.has("pause_grace_seconds")) ctl.put("pause_grace_seconds", previousControl.getInt("pause_grace_seconds"))
                    }
                    write(File(dir, "retry_origin.json"), JSONObject().put("task", id).put("mode", "reinitialize"))
                    pause(id)
                    note(next, "ready", "New initialization from $id, not exact resume. Original failure artifacts retained.")
                    next
                } catch (e: Exception) {
                    dir.deleteRecursively()
                    throw e
                }
        }) { "Pause training before creating a retry task" }
    }
    fun respectSystemStop(id: String) {
        if (Build.VERSION.SDK_INT >= 30) {
            val stop = context.getSystemService(ActivityManager::class.java)
                .getHistoricalProcessExitReasons(null, 0, 16)
                .filter { it.reason == ApplicationExitInfo.REASON_USER_REQUESTED }.maxOfOrNull { it.timestamp }
            if (stop != null) applySystemStop(id, stop)
        }
    }
    internal fun applySystemStop(id: String, timestamp: Long) = locked(id) {
        val current = readControl(id)
        if (current.optBoolean("enabled") && timestamp > current.optLong("explicit_resume_ms"))
            disableLocked(id, current)
    }
    fun importTree(uri: Uri): String {
        val tree = requireNotNull(DocumentFile.fromTreeUri(context, uri))
        val config = requireNotNull(tree.findFile("experiment.json")) { "experiment.json is missing" }
        val json = context.contentResolver.openInputStream(config.uri)!!.bufferedReader().use { it.readText() }
        val spec = JSONObject(json)
        validateSpec(spec)
        val id = UUID.randomUUID().toString()
        val dir = task(id).apply { mkdirs() }
        try {
            for (name in listOf("model/config.json", "model/model.safetensors", "train.jsonl", "valid.jsonl")) {
                var source = tree
                for (part in name.split('/')) source = requireNotNull(source.findFile(part)) { "Missing $name" }
                require(source.isFile && dir.usableSpace > source.length() + 128L * 1024 * 1024) { "Insufficient storage for $name" }
                val target = File(dir, name).apply { parentFile!!.mkdirs() }
                context.contentResolver.openInputStream(source.uri)!!.use { input ->
                    target.outputStream().use { output -> input.copyTo(output, 1024 * 1024); output.fd.sync() }
                }
            }
            write(File(dir, "spec.json"), spec)
            note(id, "ready")
            return id
        } catch (e: Exception) {
            dir.deleteRecursively()
            throw e
        }
    }
    fun createDemo(): String {
        val id = UUID.randomUUID().toString()
        val dir = task(id).apply { mkdirs() }
        fun copy(path: String) {
            val children = context.assets.list(path) ?: emptyArray()
            if (children.isNotEmpty()) children.forEach { copy("$path/$it") }
            else {
                val dest = File(dir, path.removePrefix("tiny_qwen/"))
                dest.parentFile!!.mkdirs()
                context.assets.open(path).use { input -> dest.outputStream().use { input.copyTo(it) } }
            }
        }
        copy("tiny_qwen")
        note(id, "ready", "Tiny random Qwen fixture, not a pretrained benchmark")
        return id
    }
    fun exportRecords(id: String, uri: Uri) {
        checkNotNull(withExecutionLease {
                context.contentResolver.openOutputStream(uri, "wt")!!.use { output ->
                    ZipOutputStream(output.buffered()).use { zip ->
                        val base = task(id)
                        for (file in base.walkTopDown().onEnter { dir ->
                            val path = dir.relativeTo(base).invariantSeparatorsPath
                            path != "model" && !Regex("run/sessions/[^/]+/artifacts/shards").matches(path)
                        }.filter { it.isFile }) {
                            val path = file.relativeTo(base).invariantSeparatorsPath
                            if (path.endsWith(".lock")) continue
                            zip.putNextEntry(ZipEntry(path))
                            file.inputStream().use { it.copyTo(zip, 1024 * 1024) }
                            zip.closeEntry()
                        }
                    }
                }
        }) { "Pause training before exporting" }
    }
    companion object {
        fun cpuContentionEnabled(control: JSONObject, spec: JSONObject): Boolean =
            control.optBoolean("cpu_contention_guard", spec.optBoolean("cpu_contention_guard", false))
        fun validateMicrobatch(original: Int, requested: Int) {
            require(requested in 1..original) {
                "Physical batch must be between 1 and the original batch ($original)"
            }
        }
        private val mutex = Any()
        private val leaseMutex = Any()
        private val localLeases = mutableSetOf<String>()
        fun write(file: File, json: JSONObject) {
            val atomic = AtomicFile(file)
            val stream = atomic.startWrite()
            try { stream.write(json.toString(2).toByteArray()); atomic.finishWrite(stream) }
            catch (e: Exception) { atomic.failWrite(stream); throw e }
        }
        fun validateSpec(s: JSONObject) {
            val allowed = setOf("model", "steps", "batch_size", "seq_len", "lora_r", "grad_accum", "lr",
                "min_available_mb", "slice_seconds", "threads", "activation_checkpointing",
                "memory_efficient_attention", "sharding", "shard_budget_mb", "fixture", "pause_grace_seconds",
                "cpu_contention_guard")
            require(s.keys().asSequence().all { it in allowed }) { "Unknown experiment option; see ANDROID_CONTINUITY.md" }
            require(s.get("model") is String && s.getString("model") in listOf("qwen", "gemma")) { "model must be qwen or gemma" }
            val integerFields = setOf("steps", "batch_size", "seq_len", "lora_r", "grad_accum",
                "min_available_mb", "slice_seconds", "threads", "shard_budget_mb", "pause_grace_seconds")
            for (name in integerFields.filter { s.has(it) }) {
                val value = s.get(name)
                require(value is Number) { "$name must be an integer, not a string or boolean" }
                val number = value.toDouble()
                require(number.isFinite() && number >= Int.MIN_VALUE.toDouble() &&
                    number <= Int.MAX_VALUE.toDouble() && number == number.toInt().toDouble()) {
                    "$name must be an integer in the supported range"
                }
            }
            for (name in listOf("sharding", "activation_checkpointing", "memory_efficient_attention", "cpu_contention_guard"))
                require(!s.has(name) || s.get(name) is Boolean) { "$name must be true or false" }
            require(!s.has("fixture") || s.get("fixture") is String) { "fixture must be a string" }
            for (name in listOf("steps", "batch_size", "seq_len", "lora_r", "grad_accum"))
                require(s.getInt(name) > 0) { "$name must be positive" }
            require(s.getInt("lora_r") <= Int.MAX_VALUE / 2) { "lora_r is too large for lora_alpha" }
            require(s.getInt("seq_len") >= 2) { "seq_len must allow at least one next-token target" }
            require(s.get("lr") is Number && s.getDouble("lr").isFinite() && s.getDouble("lr") > 0 &&
                s.getDouble("lr").toFloat().isFinite() && s.getDouble("lr").toFloat() > 0) {
                "lr must be a finite positive FP32 number"
            }
            require(s.optInt("min_available_mb", 512) >= 64)
            require(s.optInt("slice_seconds", 30) in 1..120)
            require(s.optInt("pause_grace_seconds", 30) in 30..420) { "Pause grace must be 30..420 seconds" }
            require(s.optInt("threads", 4) in 1..16)
            require(s.optInt("shard_budget_mb", 768) > 0)
        }
    }
}
