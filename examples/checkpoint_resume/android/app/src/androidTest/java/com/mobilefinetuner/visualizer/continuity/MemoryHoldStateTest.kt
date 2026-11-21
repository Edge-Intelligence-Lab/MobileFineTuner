package com.mobilefinetuner.visualizer.continuity

import android.content.Context
import androidx.test.core.app.ApplicationProvider
import org.junit.Assert.*
import org.junit.Test
import java.io.File

class MemoryHoldStateTest {
    private val context = ApplicationProvider.getApplicationContext<Context>()
    private val store = TaskStore(context)

    @Test fun oldPressureCannotCancelNewerStartOrOverrideManualPause() {
        val id = store.createDemo()
        val old = store.allow(id, true, false)
        val current = store.allow(id, true, false)
        assertFalse(store.holdForMemory(id, old, "old sample"))
        assertTrue(store.control(id).getBoolean("enabled"))
        store.pause(id)
        assertFalse(store.holdForMemory(id, current, "late sample"))
        assertFalse(store.control(id).has("memory_hold_reason"))
    }

    @Test fun pressureIsNotANewerUserIntentAndCannotRescheduleAnOldJob() {
        val id = store.createDemo()
        val previous = store.allow(id, true, false)
        val issued = android.os.SystemClock.elapsedRealtimeNanos()
        assertTrue(store.holdForMemory(id, previous, "pressure while UI request in flight"))
        assertFalse(store.canRetryInBackground(id, previous))
        val next = requireNotNull(store.admitStart(id, true, false, issued))
        assertFalse(store.canRetryInBackground(id, previous))
        assertTrue(store.canRetryInBackground(id, next))
        val oldClick = android.os.SystemClock.elapsedRealtimeNanos()
        store.pause(id)
        assertNull(store.admitStart(id, true, false, oldClick))
    }

    @Test fun partialFirstAttemptUnderHoldDoesNotPromiseExactResume() {
        val id = store.createDemo()
        val gen = store.allow(id, false, false)
        File(store.task(id), "attempts").mkdirs()
        File(store.task(id), "attempts/partial.log").writeText("started")
        assertTrue(store.holdForMemory(id, gen, "pressure"))
        val status = store.status(id)
        assertTrue(status.getBoolean("retry_as_new"))
        assertTrue(status.getString("detail").contains("Retry as new task"))
    }

    @Test fun pressureHoldSurvivesNewControllerAndRetriesOnlyOnExplicitAdmission() {
        val id = store.createDemo()
        val original = store.spec(id)
        TaskStore.write(File(store.task(id), "spec.json"), store.spec(id).put("min_available_mb", 1000000))
        store.allow(id, true, false)
        assertEquals(TrainingRunner.Outcome.WAITING, TrainingRunner(context).run(id, false))
        val reopened = TaskStore(context)
        assertEquals("memory_hold", reopened.status(id).getString("state"))
        assertFalse(reopened.control(id).getBoolean("enabled"))
        assertFalse(reopened.control(id).getBoolean("automatic"))
        assertFalse(store.hasAttemptArtifacts(id))
        TaskStore.write(File(store.task(id), "spec.json"), original)
        assertEquals(TrainingRunner.Outcome.DISABLED, TrainingRunner(context).run(id, true))
        assertFalse(store.hasAttemptArtifacts(id))
        reopened.allow(id, false, false)
        assertFalse(reopened.control(id).has("memory_hold_reason"))
        assertEquals(TrainingRunner.Outcome.COMPLETED, TrainingRunner(context).run(id, false))
    }
}
