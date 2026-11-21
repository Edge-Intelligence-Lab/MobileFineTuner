package com.mobilefinetuner.visualizer.continuity

import android.content.Context
import android.os.SystemClock
import android.system.Os
import android.system.OsConstants
import androidx.test.core.app.ApplicationProvider
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Test
import java.io.File
import java.util.concurrent.FutureTask
import java.util.concurrent.TimeUnit

class UserStopTest {
    private val ctx = ApplicationProvider.getApplicationContext<Context>()
    private val store = TaskStore(ctx)
    private val runner = TrainingRunner(ctx)
    private val focus = FocusWindow(ctx)

    @Test fun focusDisarmsAllQueuedWorkEvenAfterExpiryOrEarlyEnd() {
        val a = store.createDemo()
        val b = store.createDemo()
        try {
            val ga = store.allow(a, true, false)
            store.allow(b, true, false)
            val oldTap = SystemClock.elapsedRealtimeNanos()
            focus.start()
            assertTrue(FocusWindow(ctx).withGate { it.remainingMs in 1..FocusWindow.HOUR })
            assertFalse(store.canRetryInBackground(a, ga))
            assertEquals(TrainingRunner.Outcome.DISABLED, runner.run(b, true))
            assertTrue(runCatching { store.allow(b, false, false) }.isFailure)
            val during = SystemClock.elapsedRealtimeNanos()
            focus.end()
            assertNull(store.admitStart(a, false, false, issued = during))
            assertFalse(store.control(b).getBoolean("enabled"))
            assertNull(store.admitStart(a, false, false, issued = oldTap))
            assertNotNull(store.admitStart(a, false, false))
            focus.start()
            // Simulate expiry on this boot; do not wait an hour in an instrumentation test.
            val f = File(ctx.filesDir, "training_focus.json")
            val tapBeforeExpiry = SystemClock.elapsedRealtimeNanos()
            SystemClock.sleep(4)
            val endMs = SystemClock.elapsedRealtime()
            TaskStore.write(f, JSONObject(f.readText()).put("until_elapsed", endMs).put("resume_after_ns", endMs * 1000000L))
            assertEquals(0L, focus.withGate { it.remainingMs })
            assertNull(store.admitStart(a, false, false, issued = tapBeforeExpiry))
            assertFalse(store.control(a).getBoolean("enabled"))
            assertNotNull(store.admitStart(a, false, false))
        } finally {
            focus.end()
            store.task(a).deleteRecursively()
            store.task(b).deleteRecursively()
        }
    }
    @Test fun focusReanchorsOnceOnNewBoot() {
        val f = File(ctx.filesDir, "training_focus.json")
        try {
            focus.start()
            TaskStore.write(f, JSONObject(f.readText()).put("boot", -9)
                .put("until_wall", System.currentTimeMillis() + 60000L))
            assertTrue(focus.withGate { it.remainingMs } in 1L..60000L)
            // After re-anchoring, clock changes cannot extend or expire the window.
            TaskStore.write(f, JSONObject(f.readText()).put("until_wall", 1L))
            assertTrue(focus.withGate { it.remainingMs } in 1L..60000L)
        } finally { focus.end() }
    }

    private fun state(id: String): File {
        val dir = File(store.task(id), "run")
        return File(dir, "training_state/" + JSONObject(File(dir, "LATEST").readText()).getString("current"))
    }
    private fun compareAfterRelease(byFocus: Boolean) {
        val a = store.createDemo()
        val b = store.createDemo()
        try {
            for (id in listOf(a, b))
                TaskStore.write(File(store.task(id), "spec.json"), store.spec(id).put("steps", 100).put("slice_seconds", 120))
            store.allow(a, false, false)
            assertEquals(TrainingRunner.Outcome.COMPLETED, runner.run(a, false))
            store.task(a).walkTopDown().single { it.name == "lora_initial.safetensors" }
                .copyTo(File(store.task(b), "initial.safetensors"))
            val g = store.allow(b, true, false)
            val job = FutureTask { runner.run(b, true, expectedGeneration = g) }
            Thread(job).start()
            val deadline = SystemClock.elapsedRealtime() + 15000
            while ((store.checkpoint(b)?.optInt("committed_updates") ?: 0) < 1) {
                check(!job.isDone && SystemClock.elapsedRealtime() < deadline)
                SystemClock.sleep(2)
            }
            val events = File(store.task(b), "run/sessions").walkTopDown().single { it.name == "events.jsonl" }
            val pid = JSONObject(events.bufferedReader().use { it.readLine() }).getInt("pid")
            // A stopped child cannot reach the cooperative boundary: only immediate release can finish.
            Os.kill(pid, OsConstants.SIGSTOP)
            try {
                if (byFocus) focus.start() else store.pause(b, immediately = true)
                assertEquals(TrainingRunner.Outcome.DISABLED, job.get(8, TimeUnit.SECONDS))
            } finally {
                if (!job.isDone) runCatching { Os.kill(pid, OsConstants.SIGCONT) }
            }
            assertFalse(store.isExecuting(b))
            assertFalse(File("/proc/$pid").exists())
            assertTrue(store.checkpoint(b)!!.getInt("committed_updates") in 1 until 100)
            if (byFocus) focus.end()
            assertFalse(store.canRetryInBackground(b, g))
            store.allow(b, false, false)
            assertEquals(TrainingRunner.Outcome.COMPLETED, runner.run(b, false))
            val left = state(a)
            val right = state(b)
            assertArrayEquals(File(left, "state.safetensors").readBytes(), File(right, "state.safetensors").readBytes())
            val s = JSONObject(File(left, "state.json").readText())
            val t = JSONObject(File(right, "state.json").readText())
            assertEquals(s.getString("dataset_state"), t.getString("dataset_state"))
            assertEquals(s.getJSONObject("optimizer").toString(), t.getJSONObject("optimizer").toString())
        } finally {
            if (byFocus) focus.end()
            for (id in listOf(a,b)) { store.pause(id); if (!store.isExecuting(id)) store.task(id).deleteRecursively() }
        }
    }
    @Test fun releaseNowReplaysCommittedStateExactly() = compareAfterRelease(false)
    @Test fun focusReleasesLiveChildAndRequiresManualResume() = compareAfterRelease(true)
}
