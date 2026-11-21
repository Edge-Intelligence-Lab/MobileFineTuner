package com.mobilefinetuner.visualizer.continuity

import android.content.Context
import android.content.Intent
import android.net.Uri
import android.app.ActivityManager
import androidx.test.core.app.ActivityScenario
import androidx.core.content.ContextCompat
import androidx.work.WorkManager
import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.core.app.ApplicationProvider
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Test
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

class ContinuityTest {
    private val context = ApplicationProvider.getApplicationContext<Context>()
    private val store = TaskStore(context)
    private fun create(steps: Int): String {
        val id = store.createDemo()
        val spec = store.spec(id).put("steps", steps)
        TaskStore.write(File(store.task(id), "spec.json"), spec)
        store.allow(id, false, false)
        return id
    }
    private fun state(id: String): Pair<JSONObject, File> {
        val run = File(store.task(id), "run")
        val gen = JSONObject(File(run, "LATEST").readText()).getString("current")
        val dir = File(run, "training_state/$gen")
        return JSONObject(File(dir, "state.json").readText()) to dir
    }
    private fun parity(steps: Int, stop: Int, corruptSampler: Boolean = false, flags: List<String> = emptyList()) {
        fun createPairTask(): String = create(steps).also { id ->
            if (flags.isNotEmpty()) {
                val spec = store.spec(id).put("grad_accum", if ("accumulation" in flags) 2 else 1)
                for (key in flags - "accumulation") spec.put(key, true)
                TaskStore.write(File(store.task(id), "spec.json"), spec)
            }
        }
        val baseline = createPairTask()
        assertEquals(TrainingRunner.Outcome.COMPLETED, TrainingRunner(context).run(baseline, false))
        val split = createPairTask()
        val initial = File(store.task(baseline), "run/sessions").walkTopDown()
            .single { it.name == "lora_initial.safetensors" }
        initial.copyTo(File(store.task(split), "initial.safetensors"))
        assertEquals(TrainingRunner.Outcome.SUSPENDED, TrainingRunner(context).run(split, false, stopAfter = stop))
        assertEquals(stop, store.status(split).getInt("committed_updates"))
        if (corruptSampler) {
            val file = File(state(split).second, "state.json")
            val original = file.readText()
            val i = original.lastIndexOf("classic_stream_v1")
            assertTrue(i >= 0)
            file.writeText(original.replaceRange(i, i + 17, "classic_stream_v0"))
        }
        assertEquals(TrainingRunner.Outcome.COMPLETED, TrainingRunner(context).run(split, false))
        val (a, ad) = state(baseline)
        val (b, bd) = state(split)
        assertArrayEquals(File(ad, "state.safetensors").readBytes(), File(bd, "state.safetensors").readBytes())
        assertEquals(a.getJSONObject("optimizer").toString(), b.getJSONObject("optimizer").toString())
        assertEquals(a.getString("dataset_state"), b.getString("dataset_state"))
        val ta = a.getJSONObject("trainer"); val tb = b.getJSONObject("trainer")
        val bestA = File(ta.getString("best_path")); val bestB = File(tb.getString("best_path"))
        assertArrayEquals(bestA.readBytes(), bestB.readBytes())
        ta.remove("best_path"); tb.remove("best_path")
        assertEquals(steps, tb.getJSONObject("timing").getInt("updates"))
        assertTrue(tb.getJSONObject("timing").getDouble("full_step_sec") > 0)
        ta.remove("timing"); tb.remove("timing") // Wall time is measured, not deterministic.
        assertEquals(ta.toString(), tb.toString())
        assertEquals(steps, b.getJSONObject("trainer").getInt("completed_updates"))
        val events = File(store.task(split), "run/sessions").walkTopDown().filter { it.name == "events.jsonl" }
            .flatMap { f -> f.readLines().map { JSONObject(it) }.asSequence() }.toList()
        val ready = events.single { it.getString("type") == "resume_ready" }.getJSONObject("metrics")
        if (corruptSampler) {
            val resume = events.single { it.getString("type") == "resume" }
            assertTrue(resume.getBoolean("fallback"))
            assertEquals(stop - 1, resume.getInt("completed_updates"))
        }
        assertTrue(ready.getDouble("wall_sec") > 0)
        assertTrue(ready.getLong("peak_sampled_rss_bytes") > 0)
        assertTrue(events.any { (it.optJSONObject("save_metrics")?.optLong("logical_write_bytes", 0) ?: 0) > 0 })
        TaskStore.write(File(store.task(split), "parity_result.json"), JSONObject()
            .put("baseline", baseline).put("resumed", split).put("steps", steps).put("split", stop)
            .put("exact_tensor_optimizer_dataset", true))
    }
    @Test fun resume10AcrossNativeProcesses() = parity(10, 4)
    @Test fun invalidSamplerFallsBackAndReplaysExactly() = parity(10, 4, true)
    @Test fun oneReplayedUpdateCountsEvenAtOldHighWatermark() {
        val id = create(10)
        store.allow(id, true, false)
        val runner = TrainingRunner(context)
        assertEquals(TrainingRunner.Outcome.SUSPENDED, runner.run(id, true, stopAfter = 4))
        val file = File(state(id).second, "state.json")
        val content = file.readText()
        val i = content.lastIndexOf("classic_stream_v1")
        assertTrue(i >= 0)
        file.writeText(content.replaceRange(i, i + 17, "classic_stream_v0"))
        assertEquals(4, store.status(id).getInt("committed_updates"))
        assertEquals(TrainingRunner.Outcome.SUSPENDED, runner.run(id, true, stopAfter = 4))
        val s = store.status(id)
        val actualStart = TrainingRunner.sessionStartUpdate(File(store.task(id), "run"), s.getString("committing_session"))
        assertEquals(3, actualStart)
        assertEquals(4, s.getInt("committed_updates"))
        assertFalse(TrainingRunner.exhaustedWithoutProgress(actualStart, 4, 30000, 30, false))
        assertTrue(store.control(id).getBoolean("automatic"))
        assertEquals(TrainingRunner.Outcome.COMPLETED, runner.run(id, true))
    }
    @Test fun tinyIndividualAndCombinedMemoryModesResumeExactly() {
        val modes = listOf("accumulation", "activation_checkpointing", "memory_efficient_attention", "sharding")
        for (flags in modes.map { listOf(it) } + listOf(modes)) parity(10, 4, flags = flags)
    }
    @Test fun malformedRowCannotSilentlyShortenTaskDataset() {
        val id = create(2)
        File(store.task(id), "train.jsonl").appendText("\n{\"ids\":[0,1],\"mask\":[0]}\n")
        assertEquals(TrainingRunner.Outcome.FAILED, TrainingRunner(context).run(id, false))
        assertFalse(File(store.task(id), "run/LATEST").exists())
        assertTrue(store.status(id).getString("detail").contains("train.jsonl:"))
    }
    @Test fun resume200AcrossNativeProcesses() = parity(200, 73)
    @Test fun manualPausePreventsScheduledRun() {
        val id = create(10)
        store.allow(id, true, false)
        store.pause(id)
        assertEquals(TrainingRunner.Outcome.DISABLED, TrainingRunner(context).run(id, true))
        assertFalse(File(store.task(id), "run/LATEST").exists())
    }
    @Test fun executionLeasePreventsDuplicateChild() {
        val id = create(10)
        assertNotNull(store.withExecutionLease {
            assertEquals(TrainingRunner.Outcome.BUSY, TrainingRunner(context).run(id, false))
        })
    }
    @Test fun memoryGateDoesNotLaunchOrChangeBatch() {
        val id = create(10)
        TaskStore.write(File(store.task(id), "spec.json"), store.spec(id).put("min_available_mb", 1000000))
        assertEquals(TrainingRunner.Outcome.WAITING, TrainingRunner(context).run(id, false))
        assertEquals(2, store.spec(id).getInt("batch_size"))
        assertFalse(File(store.task(id), "run/LATEST").exists())
    }
    private fun awaitCondition(message: String, timeoutMs: Long = 60000, predicate: () -> Boolean) {
        val end = android.os.SystemClock.elapsedRealtime() + timeoutMs
        while (android.os.SystemClock.elapsedRealtime() < end) {
            if (predicate()) return
            Thread.sleep(100)
        }
        fail(message)
    }
    private fun startService(id: String) {
        ContextCompat.startForegroundService(context, Intent(context, TrainingService::class.java).putExtra("task", id))
    }
    private fun leaseFree(): Boolean = !store.executionBusy()
    @Suppress("DEPRECATION")
    private fun serviceStopped(): Boolean = context.getSystemService(ActivityManager::class.java)
        .getRunningServices(Int.MAX_VALUE).none { it.service.className == TrainingService::class.java.name }
    @Test fun foregroundPauseAndContinue() {
        ActivityScenario.launch(TrainingActivity::class.java).use {
            val id = create(2000)
            startService(id)
            awaitCondition("service failed to train: ${store.status(id)}") { store.status(id).optInt("committed_updates") >= 2 }
            store.pause(id)
            awaitCondition("service failed to suspend") { store.status(id).optString("state") == "suspended" && leaseFree() }
            val paused = store.status(id).getInt("committed_updates")
            Thread.sleep(1200)
            assertEquals(paused, store.status(id).getInt("committed_updates"))
            assertFalse(store.control(id).getBoolean("automatic"))
            Thread.sleep(500)
            store.allow(id, false, false)
            startService(id)
            awaitCondition("service failed to resume") { store.status(id).optInt("committed_updates") > paused }
            store.pause(id)
            awaitCondition("resume service failed to suspend") { store.status(id).optString("state") == "suspended" && leaseFree() }
        }
    }
    @Test fun workerContinuesFiniteSlices() {
        val id = create(400)
        TaskStore.write(File(store.task(id), "spec.json"), store.spec(id).put("slice_seconds", 1))
        Scheduling.schedule(context, id, store.allow(id, true, false))
        try {
            awaitCondition("worker did not finish: ${store.status(id)}", 180000) {
                store.status(id).optString("committed_status") == "completed" && leaseFree()
            }
            assertEquals(400, store.status(id).getInt("committed_updates"))
            assertTrue(File(store.task(id), "run/sessions").listFiles()!!.size > 1)
        } finally { Scheduling.cancel(context, id, store.pause(id)) }
    }
    @Test fun parentDeathKillsNativeAndColdResumeWorks() {
        ActivityScenario.launch(TrainingActivity::class.java).use {
            val id = create(2000)
            startService(id)
            awaitCondition("no progress before process kill") { store.status(id).optInt("committed_updates") >= 2 }
            val eventFile = File(store.task(id), "run/sessions").walkTopDown().first { it.name == "events.jsonl" }
            val pid = JSONObject(eventFile.readLines().first()).getInt("pid")
            val process = context.getSystemService(ActivityManager::class.java).runningAppProcesses
                .single { it.processName == context.packageName + ":training" }
            android.os.Process.killProcess(process.pid)
            awaitCondition("native orphan remains after parent death") { !File("/proc/$pid").exists() && leaseFree() }
            val last = store.status(id).getInt("committed_updates")
            assertTrue(last >= 2)
            store.allow(id, false, false)
            val result = TrainingRunner(context).run(id, false, stopAfter = last + 2)
            assertEquals(TrainingRunner.Outcome.SUSPENDED, result)
            assertEquals(last + 2, store.status(id).getInt("committed_updates"))
            store.pause(id)
        }
    }
    @Test fun screenOffKeepsFiniteForegroundSlice() {
        ActivityScenario.launch(TrainingActivity::class.java).use {
            val id = create(2000)
            startService(id)
            awaitCondition("no progress before screen off") { store.status(id).optInt("committed_updates") >= 2 }
            val before = store.status(id).getInt("committed_updates")
            val automation = InstrumentationRegistry.getInstrumentation().uiAutomation
            try {
                automation.executeShellCommand("input keyevent 223").close()
                awaitCondition("no training progress with screen off") { store.status(id).optInt("committed_updates") > before + 10 }
                store.pause(id)
                awaitCondition("no screen-off suspension") { store.status(id).optString("state") == "suspended" && leaseFree() }
            } finally { automation.executeShellCommand("input keyevent 224").close() }
        }
    }
    @Test fun cancelledWorkerCannotReenableManualPause() {
        val id = create(2000)
        Scheduling.schedule(context, id, store.allow(id, true, false))
        try {
            awaitCondition("no worker progress") { store.status(id).optInt("committed_updates") >= 2 }
            Scheduling.cancel(context, id, store.pause(id))
            awaitCondition("cancelled worker still owns training") { leaseFree() }
            Thread.sleep(1000)
            assertFalse(store.control(id).getBoolean("enabled"))
            assertFalse(store.control(id).getBoolean("automatic"))
            val last = store.status(id).getInt("committed_updates")
            store.allow(id, false, false)
            assertEquals(TrainingRunner.Outcome.SUSPENDED, TrainingRunner(context).run(id, false, stopAfter = last + 2))
        } finally { Scheduling.cancel(context, id, store.pause(id)) }
    }
    @Test fun malformedConfigFailsPreflightButCanRetrySamePristineTask() {
        val id = create(10)
        val config = File(store.task(id), "model/config.json")
        val original = config.readBytes()
        try {
            config.writeText("{invalid config")
            assertEquals(TrainingRunner.Outcome.FAILED, TrainingRunner(context).run(id, false))
            val status = store.status(id)
            assertEquals("failed", status.getString("state"))
            assertFalse(status.getBoolean("checkpoint_available"))
            assertFalse(status.getBoolean("retry_as_new"))
            assertFalse(store.hasAttemptArtifacts(id))
            assertTrue(status.getString("detail").contains("same task"))
        } finally { config.writeBytes(original) }
        store.allow(id, false, false)
        assertEquals(TrainingRunner.Outcome.SUSPENDED, TrainingRunner(context).run(id, false, stopAfter = 2))
        store.pause(id)
    }
    @Test fun missingPrivateWeightFailsBeforeCommitAndRetainsRetryEvidence() {
        val id = create(10)
        val weights = File(store.task(id), "model/model.safetensors")
        val retained = File(store.task(id), "model/retained-for-missing-weight-test.safetensors")
        assertTrue(weights.renameTo(retained))
        try {
            assertEquals(TrainingRunner.Outcome.FAILED, TrainingRunner(context).run(id, false))
            val status = store.status(id)
            assertFalse(File(store.task(id), "run/LATEST").exists())
            assertFalse(status.getBoolean("checkpoint_available"))
            assertTrue(status.getBoolean("retry_as_new"))
            assertFalse(store.control(id).getBoolean("enabled"))
            val log = File(store.task(id), "attempts").listFiles()!!.single()
            assertTrue(log.readText().contains("model.safetensors"))
            assertTrue(status.getString("detail").contains("model.safetensors"))
        } finally { assertTrue(retained.renameTo(weights)) }
        val failure = File(store.task(id), "ui_state.json").readBytes()
        val next = store.retryAsNew(id)
        assertFalse(store.control(next).getBoolean("enabled"))
        assertArrayEquals(failure, File(store.task(id), "ui_state.json").readBytes())
        assertTrue(File(store.task(id), "attempts").listFiles()!!.isNotEmpty())
    }
    @Test fun incompleteBackboneIsRejectedBeforeFirstCommit() {
        val id = create(10)
        val weights = File(store.task(id), "model/model.safetensors")
        val original = weights.readBytes()
        val headerSize = ByteBuffer.wrap(original, 0, 8).order(ByteOrder.LITTLE_ENDIAN).long.toInt()
        val header = JSONObject(String(original, 8, headerSize, Charsets.UTF_8))
        val missing = "model.layers.0.self_attn.q_proj.weight"
        assertNotNull(header.remove(missing))
        val encoded = header.toString().toByteArray()
        val padded = encoded + ByteArray((8 - encoded.size % 8) % 8) { 32 }
        val length = ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN).putLong(padded.size.toLong()).array()
        try {
            weights.writeBytes(length + padded + original.copyOfRange(8 + headerSize, original.size))
            assertEquals(TrainingRunner.Outcome.FAILED, TrainingRunner(context).run(id, false))
            assertFalse(File(store.task(id), "run/LATEST").exists())
            assertFalse(store.status(id).getBoolean("checkpoint_available"))
            assertTrue(store.status(id).getString("detail").contains("Missing Qwen base tensor: $missing"))
        } finally { weights.writeBytes(original) }
        val next = store.retryAsNew(id)
        store.allow(next, false, false)
        assertEquals(TrainingRunner.Outcome.SUSPENDED, TrainingRunner(context).run(next, false, stopAfter = 4))
        assertEquals(TrainingRunner.Outcome.COMPLETED, TrainingRunner(context).run(next, false))
        assertEquals(10, store.status(next).getInt("committed_updates"))
    }
    @Test fun invalidSpecFailsBeforeNativeAttempt() {
        val id = create(10)
        val original = store.spec(id)
        for ((key, value) in listOf("sharding" to 1, "activation_checkpointing" to "on",
                "threads" to "oops", "steps" to 2.5)) {
            TaskStore.write(File(store.task(id), "spec.json"), JSONObject(original.toString()).put(key, value))
            store.allow(id, false, false)
            assertEquals(TrainingRunner.Outcome.FAILED, TrainingRunner(context).run(id, false))
            assertFalse(store.hasAttemptArtifacts(id))
            assertFalse(File(store.task(id), "run/LATEST").exists())
        }
        TaskStore.write(File(store.task(id), "spec.json"), original)
        store.pause(id)
    }
    @Test fun cancelledOldRunnerExceptionCannotRevokeNewResume() {
        val id = create(5000)
        var newGeneration: Long? = null
        try {
            val outcome = TrainingRunner(context).run(id, false, cancelled = {
                if (store.hasAttemptArtifacts(id) && newGeneration == null) {
                    newGeneration = store.allow(id, true, false)
                    store.note(id, "starting", "newer user Resume")
                    throw IllegalStateException("old slice cleanup failed")
                }
                false
            })
            assertNotNull(newGeneration)
            assertEquals(TrainingRunner.Outcome.FAILED, outcome)
            assertEquals(checkNotNull(newGeneration), store.control(id).getLong("resume_generation"))
            assertTrue(store.control(id).getBoolean("enabled"))
            assertTrue(store.control(id).getBoolean("automatic"))
            assertEquals("newer user Resume", store.status(id).getString("historical_detail"))
            assertTrue(File(store.task(id), "last_stale_failure.json").readText().contains("old slice cleanup failed"))
        } finally { store.pause(id) }
    }
    @Test fun rejectedLocalDuplicateAndExportLeaveKernelLeaseHeldAgainstService() {
        val id = create(10)
        try {
            ActivityScenario.launch(TrainingActivity::class.java).use {
                awaitCondition("previous service still active before kernel lease check", 10000) { serviceStopped() }
                assertNotNull(store.withExecutionLease {
                    assertEquals(TrainingRunner.Outcome.BUSY, TrainingRunner(context).run(id, false))
                    assertTrue(store.executionBusy())
                    val exportError = runCatching { store.exportRecords(id, Uri.parse("content://unused/lease-test")) }.exceptionOrNull()
                    assertTrue(exportError is IllegalStateException)
                    assertEquals("Pause training before exporting", exportError?.message)
                    store.allow(id, false, false)
                    store.note(id, "starting", "cross-process lease check")
                    // TrainingService runs in :training, so its rejection checks the kernel lease.
                    startService(id)
                    awaitCondition("service acquired a lease lost by the rejected local call", 10000) {
                        store.status(id).optString("state") == "rejected"
                    }
                    assertFalse(store.hasAttemptArtifacts(id))
                    assertTrue(store.executionBusy())
                    awaitCondition("lease-check service did not stop", 10000) { serviceStopped() }
                })
            }
        } finally {
            store.pause(id)
            awaitCondition("service still owns execution after lease test") { leaseFree() }
        }
    }
    @Test fun oldNativeNonzeroExitCannotRevokeNewResume() {
        val id = create(5000)
        var newGeneration: Long? = null
        try {
            val outcome = TrainingRunner(context).run(id, false, cancelled = {
                if (newGeneration == null && (store.checkpoint(id)?.optInt("committed_updates") ?: 0) >= 1) {
                    val events = File(store.task(id), "run/sessions").walkTopDown().first { it.name == "events.jsonl" }
                    val pid = JSONObject(events.readLines().first()).getInt("pid")
                    newGeneration = store.allow(id, true, false)
                    store.note(id, "starting", "new Resume before old native failure")
                    android.os.Process.killProcess(pid)
                }
                false
            })
            assertNotNull(newGeneration)
            assertEquals(TrainingRunner.Outcome.FAILED, outcome)
            assertEquals(checkNotNull(newGeneration), store.control(id).getLong("resume_generation"))
            assertTrue(store.control(id).getBoolean("enabled"))
            assertTrue(store.control(id).getBoolean("automatic"))
            assertEquals("new Resume before old native failure", store.status(id).getString("historical_detail"))
            assertTrue(File(store.task(id), "last_stale_failure.json").readText().contains("deadline=false"))
        } finally { store.pause(id) }
    }
    @Test fun failureAfterCommitRetainsResumeAndDiagnostics() {
        val id = create(10)
        assertEquals(TrainingRunner.Outcome.SUSPENDED, TrainingRunner(context).run(id, false, stopAfter = 2))
        val checkpoint = state(id).second
        val tensors = File(checkpoint, "state.safetensors").readBytes()
        File(store.task(id), "train.jsonl").writeText("invalid input\n")
        assertEquals(TrainingRunner.Outcome.FAILED, TrainingRunner(context).run(id, false))
        val status = store.status(id)
        assertEquals("failed", status.getString("state"))
        assertTrue(status.getBoolean("checkpoint_available"))
        assertFalse(status.getBoolean("retry_as_new"))
        assertTrue(status.getString("detail").contains("Start / Resume validates"))
        assertArrayEquals(tensors, File(checkpoint, "state.safetensors").readBytes())
        assertTrue(File(store.task(id), "attempts").listFiles()!!.size >= 2)
    }
}
