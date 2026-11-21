package com.mobilefinetuner.visualizer.continuity

import android.content.Context
import android.os.SystemClock
import androidx.test.core.app.ApplicationProvider
import androidx.work.OneTimeWorkRequestBuilder
import androidx.work.WorkInfo
import androidx.work.WorkManager
import androidx.work.workDataOf
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withTimeout
import org.junit.After
import org.junit.Assert.*
import org.junit.Test

class GenerationSchedulingTest {
    private val context = ApplicationProvider.getApplicationContext<Context>()
    private val store = TaskStore(context)
    private val id = store.createDemo()
    private val wm = WorkManager.getInstance(context)
    private fun pending() = wm.getWorkInfosForUniqueWork("mft-$id").get().filterNot { it.state.isFinished }
    @After fun clean() { Scheduling.cancel(context, id, store.pause(id)) }

    @Test fun delayedActivityMutationsCannotReplaceOrCancelNewAutomaticWorker() {
        // Keep the native runner out of the test even if the WM initial delay expires.
        assertNotNull(store.withExecutionLease {
            val first = store.allow(id, true, true)
            assertTrue(Scheduling.schedule(context, id, first))
            val manual = store.allow(id, false, false)
            val paused = store.pause(id)
            val auto = store.allow(id, true, false)
            assertTrue(Scheduling.schedule(context, id, auto, replace = true))
            val winner = pending().single()
            assertFalse(winner.constraints.requiresCharging())
            assertTrue(Scheduling.tag(auto) in winner.tags)

            // Resume, manual handoff, and Pause effects from older Activity callbacks arrive last.
            assertFalse(Scheduling.schedule(context, id, first, replace = true))
            assertFalse(Scheduling.cancel(context, id, manual))
            assertFalse(Scheduling.cancel(context, id, paused))
            assertFalse(store.pauseIfCurrent(id, first))
            assertFalse(store.dispatchStartIfCurrent(id, first) { error("stale foreground dispatch") })
            assertTrue(Scheduling.schedule(context, id, auto)) // boot KEEP
            assertEquals(winner.id, pending().single().id)

            val charging = store.allow(id, true, true)
            assertTrue(Scheduling.schedule(context, id, charging)) // boot with obsolete constraints/gen
            val changed = pending().single()
            assertNotEquals(winner.id, changed.id)
            assertTrue(changed.constraints.requiresCharging())
            assertTrue(Scheduling.tag(charging) in changed.tags)
            val lastManual = store.allow(id, false, false)
            assertTrue(Scheduling.cancel(context, id, lastManual))
            assertTrue(pending().isEmpty())
        })
    }

    @Test fun pauseInvalidatesQueuedResumeAndSameGenerationDispatch() {
        val gen = store.allow(id, true, true)
        val queued = SystemClock.elapsedRealtimeNanos()
        assertEquals(gen, store.pause(id))
        assertNull(store.admitStart(id, true, false, issued = queued))
        assertFalse(Scheduling.schedule(context, id, gen, replace = true))
        assertFalse(store.dispatchStartIfCurrent(id, gen) { error("dispatch after Pause") })
        assertTrue(pending().isEmpty())
    }

    @Test fun staleAndUnversionedWorkersExitWithoutAdoptingCurrentGeneration() {
        val old = store.allow(id, true, false)
        val current = store.allow(id, true, false)
        for (data in listOf(workDataOf("task" to id), workDataOf("task" to id, Scheduling.GENERATION to old))) {
            val request = OneTimeWorkRequestBuilder<TrainingWorker>().setInputData(data).build()
            try {
                wm.enqueue(request).result.get()
                val terminal = runBlocking {
                    withTimeout(60000L) { wm.getWorkInfoByIdFlow(request.id).first { it?.state?.isFinished == true }!! }
                }
                assertEquals(WorkInfo.State.SUCCEEDED, terminal.state)
                assertFalse(store.hasAttemptArtifacts(id))
                assertTrue(store.canRetryInBackground(id, current))
            } finally { wm.cancelWorkById(request.id).result.get() }
        }
    }
}
