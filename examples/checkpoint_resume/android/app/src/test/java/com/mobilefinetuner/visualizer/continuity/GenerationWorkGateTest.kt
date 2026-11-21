package com.mobilefinetuner.visualizer.continuity

import org.junit.Assert.*
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.util.concurrent.CountDownLatch
import java.util.concurrent.FutureTask
import java.util.concurrent.TimeUnit

class GenerationWorkGateTest {
    @get:Rule val temp = TemporaryFolder()

    @Test fun inFlightCancelCompletesBeforeNewEnqueueWithoutBlockingControl() {
        val task = temp.newFolder()
        val controlLock = Any()
        var policy = WorkPolicy(1, false, true)
        fun read() = synchronized(controlLock) { policy }
        val mutations = mutableListOf<String>()
        val cancelling = CountDownLatch(1)
        val finish = CountDownLatch(1)
        val cancel = FutureTask {
            GenerationWorkGate.apply(task, 1, false, ::read) {
                cancelling.countDown()
                check(finish.await(10, TimeUnit.SECONDS))
                // Simulate a completing worker re-entering task control.
                assertEquals(2L, read().generation)
                mutations.add("cancel 1")
            }
        }
        Thread(cancel).start()
        try {
            assertTrue(cancelling.await(5, TimeUnit.SECONDS))
            synchronized(controlLock) { policy = WorkPolicy(2, true, false) }
            val enqueue = FutureTask {
                GenerationWorkGate.apply(task, 2, true, ::read) { mutations.add("enqueue 2") }
            }
            val contender = Thread(enqueue).apply { start() }
            val end = System.nanoTime() + TimeUnit.SECONDS.toNanos(5)
            while (contender.isAlive && contender.state != Thread.State.BLOCKED && System.nanoTime() < end)
                Thread.yield()
            assertEquals(Thread.State.BLOCKED, contender.state)
            finish.countDown()
            assertTrue(enqueue.get(5, TimeUnit.SECONDS))
            assertTrue(cancel.get(5, TimeUnit.SECONDS))
            assertEquals(listOf("cancel 1", "enqueue 2"), mutations)
            assertFalse(GenerationWorkGate.apply(task, 1, false, ::read) { error("stale cancel") })
            assertFalse(GenerationWorkGate.apply(task, 1, true, ::read) { error("stale enqueue") })
        } finally { finish.countDown() }
    }

    @Test fun sameGenerationPolicyIsRecheckedAfterPause() {
        val task = temp.newFolder()
        val disabled = { WorkPolicy(7, false, true) }
        assertFalse(GenerationWorkGate.apply(task, 7, true, disabled) { error("scheduled after Pause") })
        assertTrue(GenerationWorkGate.apply(task, 7, false, disabled) {})
    }
}
