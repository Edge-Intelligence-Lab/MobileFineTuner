package com.mobilefinetuner.visualizer.continuity

import android.content.Context
import androidx.test.core.app.ApplicationProvider
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Test
import java.io.File

class TaskFlowTest {
    private val store = TaskStore(ApplicationProvider.getApplicationContext<Context>())
    @Test fun pauseSupersedesAnAlreadyIssuedButNotAdmittedStart() {
        val id = store.createDemo()
        val issued = android.os.SystemClock.elapsedRealtimeNanos()
        val release = java.util.concurrent.CountDownLatch(1)
        val pending = java.util.concurrent.FutureTask<Long?> {
            check(release.await(5, java.util.concurrent.TimeUnit.SECONDS))
            store.admitStart(id, true, false, issued, 100, 90)
        }
        Thread(pending).start()
        try { store.pause(id) } finally { release.countDown() }
        assertNull(pending.get(5, java.util.concurrent.TimeUnit.SECONDS))
        val ctl = store.control(id)
        assertFalse(ctl.optBoolean("enabled"))
        assertFalse(ctl.has("slice_seconds"))
        assertEquals(TrainingRunner.Outcome.DISABLED, TrainingRunner(store.context).run(id, true))
        assertFalse(File(store.task(id), "run/sessions").exists())
        assertNotNull(store.admitStart(id, false, false, android.os.SystemClock.elapsedRealtimeNanos()))
    }
    @Test fun stopObservedBeforeExplicitResumeCannotDisableNewGeneration() {
        val id = store.createDemo()
        store.allow(id, true, false)
        val observedStop = store.control(id).getLong("explicit_resume_ms") + 1
        Thread.sleep(15)
        store.allow(id, true, false)
        File(store.task(id), "run").mkdirs()
        store.applySystemStop(id, observedStop)
        assertTrue(store.control(id).getBoolean("enabled"))
        assertFalse(File(store.task(id), "run/SUSPEND").exists())
        store.applySystemStop(id, store.control(id).getLong("explicit_resume_ms") + 1)
        assertFalse(store.control(id).getBoolean("enabled"))
        assertTrue(File(store.task(id), "run/SUSPEND").isFile)
    }
    @Test fun lastSelectionSurvivesStoreRecreationAndMissingTask() {
        val ids = listOf(store.createDemo(), store.createDemo()).sorted()
        store.rememberSelected(ids.first())
        val fresh = TaskStore(store.context)
        assertEquals(ids.first(), fresh.lastSelected(ids))
        assertEquals(ids.last(), fresh.lastSelected(ids.drop(1)))
        assertNull(fresh.lastSelected(emptyList()))
    }
    @Test fun repeatedSuspendPollsDoNotRewriteMarkerButRestoreConsumedMarker() {
        val id = store.createDemo()
        val marker = File(store.task(id), "run/SUSPEND")
        marker.parentFile!!.mkdirs()
        store.requestSuspend(id)
        assertTrue(marker.isFile)
        assertTrue(marker.setLastModified(1600000000000L))
        val stamp = marker.lastModified()
        store.requestSuspend(id)
        assertEquals(stamp, marker.lastModified())
        assertTrue(marker.delete())
        store.requestSuspend(id)
        assertTrue(marker.isFile)
    }
    @Test fun exportAfterKilledShardSessionSkipsCacheButRetainsState() {
        val id = store.createDemo()
        val files = listOf("run/sessions/s1/artifacts/shards/frozen.bin", "run/training_state/g1/state.json")
        files.forEach { File(store.task(id), it).apply { parentFile!!.mkdirs(); writeText("fixture") } }
        val zip = File.createTempFile("export", ".zip", store.context.cacheDir)
        try {
            store.exportRecords(id, android.net.Uri.fromFile(zip))
            java.util.zip.ZipFile(zip).use {
                assertNull(it.getEntry(files.first()))
                assertNotNull(it.getEntry(files.last()))
                assertNull(it.getEntry("model/model.safetensors"))
            }
        } finally { zip.delete() }
    }
    private fun checkpoint(id: String, generation: String, updates: Int): File {
        val dir = File(store.task(id), "run/training_state/$generation").apply { mkdirs() }
        TaskStore.write(File(dir, "state.json"), JSONObject().put("schema", 1).put("status", "suspended")
            .put("trainer", JSONObject().put("completed_updates", updates).put("total_steps", 10)))
        // Deliberately a metadata fixture, not a native-valid tensor payload.
        File(dir, "state.safetensors").writeText("fixture")
        TaskStore.write(File(dir, "manifest.json"), JSONObject().put("schema", 1).put("generation", generation)
            .put("completed_updates", updates).put("files", JSONObject()
                .put("state.json", File(dir, "state.json").length())
                .put("state.safetensors", File(dir, "state.safetensors").length())))
        return dir
    }
    @Test fun waitingPauseUsesControlNotStaleHistory() {
        val id = store.createDemo()
        store.allow(id, true, true)
        store.note(id, "waiting", "waiting for charging")
        store.pause(id)
        val status = store.status(id)
        assertEquals("paused", status.getString("state"))
        assertEquals("waiting", status.getString("historical_state"))
        assertFalse(status.getJSONObject("control").getBoolean("automatic"))
        assertFalse(status.getBoolean("execution_active"))
    }
    @Test fun anotherTasksLeaseDoesNotMakeStaleRunningLive() {
        val a = store.createDemo()
        val b = store.createDemo()
        store.allow(b, false, false)
        store.note(b, "running", "old process")
        assertNotNull(store.withExecutionLease {
                store.withActiveExecution(a) {
                    assertTrue(store.status(a).getBoolean("execution_active"))
                    assertFalse(store.status(b).getBoolean("execution_active"))
                    assertEquals("interrupted", store.status(b).getString("state"))
                    assertTrue(store.anotherTaskExecuting(b))
                    assertFalse(store.anotherTaskExecuting(a))
                }
        })
        assertFalse(store.status(a).getBoolean("execution_active"))
    }
    @Test fun metadataFallbackDoesNotTreatBareLatestAsCheckpoint() {
        val id = store.createDemo()
        val current = checkpoint(id, "g-current", 4)
        checkpoint(id, "g-previous", 3)
        TaskStore.write(File(store.task(id), "run/LATEST"), JSONObject().put("schema", 1)
            .put("current", "g-current").put("previous", "g-previous"))
        assertEquals(4, store.status(id).getInt("committed_updates"))
        File(current, "state.safetensors").writeText("truncated")
        assertEquals(3, store.status(id).getInt("committed_updates"))
        assertTrue(store.status(id).getBoolean("checkpoint_fallback"))
        File(store.task(id), "run/training_state/g-previous/state.json").writeText("broken")
        store.fail(id, "Invalid checkpoint")
        val status = store.status(id)
        assertFalse(status.getBoolean("checkpoint_available"))
        assertFalse(status.has("committed_updates"))
        // The pointer is not a checkpoint, but unreadable metadata is not a native verdict either.
        assertFalse(status.getBoolean("retry_as_new"))
        assertTrue(runCatching { store.retryAsNew(id) }.exceptionOrNull() is IllegalStateException)

        val generation = store.allow(id, false, false)
        val pointer = File(store.task(id), "run/LATEST").readText()
        assertTrue(store.fail(id, "Both generations invalid", generation, exhaustedPointer = pointer))
        val exhausted = store.status(id)
        assertFalse(exhausted.getBoolean("checkpoint_available"))
        assertFalse(exhausted.has("committed_updates"))
        assertTrue(exhausted.getBoolean("retry_as_new"))
        assertTrue(exhausted.getString("detail").contains("Retry as new task"))
    }
    @Test fun retryCopiesInputsButRetainsOriginalFailureAndStartsDisabled() {
        val id = store.createDemo()
        val originalSpec = File(store.task(id), "spec.json").readBytes()
        val failedLog = File(store.task(id), "attempts/failed.log").apply { parentFile!!.mkdirs(); writeText("load failed") }
        store.updateControl(id) { it.put("slice_seconds", 60).put("pause_grace_seconds", 240) }
        store.fail(id, "load failed")
        val stateBefore = File(store.task(id), "ui_state.json").readBytes()
        val next = store.retryAsNew(id)
        assertNotEquals(id, next)
        assertArrayEquals(originalSpec, File(store.task(next), "spec.json").readBytes())
        assertArrayEquals(File(store.task(id), "model/model.safetensors").readBytes(), File(store.task(next), "model/model.safetensors").readBytes())
        assertEquals("load failed", failedLog.readText())
        assertArrayEquals(stateBefore, File(store.task(id), "ui_state.json").readBytes())
        assertFalse(File(store.task(next), "run").exists())
        assertFalse(File(store.task(next), "attempts").exists())
        assertFalse(store.control(next).getBoolean("enabled"))
        assertFalse(store.control(next).getBoolean("automatic"))
        assertEquals(240, store.control(next).getInt("pause_grace_seconds"))
        assertEquals(id, JSONObject(File(store.task(next), "retry_origin.json").readText()).getString("task"))
    }
    @Test fun checkpointTaskKeepsResumeAndCannotBeRetriedAsNew() {
        val id = store.createDemo()
        checkpoint(id, "g-current", 2)
        TaskStore.write(File(store.task(id), "run/LATEST"), JSONObject().put("schema", 1)
            .put("current", "g-current").put("previous", ""))
        store.fail(id, "Missing asset")
        assertFalse(store.status(id).getBoolean("retry_as_new"))
        assertTrue(store.status(id).getString("detail").contains("Start / Resume"))
        assertTrue(runCatching { store.retryAsNew(id) }.exceptionOrNull() is IllegalStateException)
    }
    @Test fun staleExceptionNonzeroAndDeadlineFailuresDoNotChangeNewerControl() {
        for (reason in listOf("old slice I/O exception", "exit=1, deadline=false", "exit=137, deadline=true")) {
            val id = store.createDemo()
            val oldGeneration = store.allow(id, false, false)
            val newGeneration = store.allow(id, true, true)
            assertTrue(newGeneration > oldGeneration)
            store.note(id, "starting", "new Resume")
            val control = File(store.task(id), "control.json").readBytes()
            val uiState = File(store.task(id), "ui_state.json").readBytes()
            File(store.task(id), "run").mkdirs()
            assertFalse(store.fail(id, reason, oldGeneration))
            assertArrayEquals(control, File(store.task(id), "control.json").readBytes())
            assertArrayEquals(uiState, File(store.task(id), "ui_state.json").readBytes())
            assertFalse(File(store.task(id), "run/SUSPEND").exists())
            assertEquals(reason, JSONObject(File(store.task(id), "last_stale_failure.json").readText()).getString("detail"))
            assertTrue(store.fail(id, reason, newGeneration))
            assertFalse(store.control(id).getBoolean("enabled"))
        }
    }
    @Test fun pristineAdmissionFailureAndInterruptedStartKeepSameTaskRetry() {
        val id = store.createDemo()
        val generation = store.allow(id, false, false)
        store.fail(id, "foreground admission failed", generation)
        assertFalse(store.hasAttemptArtifacts(id))
        assertFalse(store.status(id).getBoolean("retry_as_new"))
        assertTrue(store.status(id).getString("detail").contains("same task"))
        store.allow(id, false, false)
        store.note(id, "starting", "start interrupted before child launch")
        assertEquals("interrupted", store.status(id).getString("state"))
        assertFalse(store.status(id).getBoolean("retry_as_new"))
        File(store.task(id), "attempts/attempt.log").apply { parentFile!!.mkdirs(); writeText("native attempt") }
        assertTrue(store.status(id).getBoolean("retry_as_new"))
        store.pause(id)
    }
    @Test fun everyOldRunnerNotePreservesNewerHandoffState() {
        val id = store.createDemo()
        val oldGeneration = store.allow(id, false, false)
        val next = store.allow(id, false, false)
        assertTrue(store.note(id, "handoff", "new manual Resume", next))
        val before = File(store.task(id), "ui_state.json").readBytes()
        for (state in listOf("running", "waiting", "suspending", "suspended", "completed")) {
            assertFalse(store.note(id, state, "old slice", oldGeneration))
            assertArrayEquals(before, File(store.task(id), "ui_state.json").readBytes())
        }
        assertTrue(store.note(id, "running", "new slice", next))
        assertEquals(next, store.status(id).getLong("resume_generation"))
        store.pause(id)
    }
}
