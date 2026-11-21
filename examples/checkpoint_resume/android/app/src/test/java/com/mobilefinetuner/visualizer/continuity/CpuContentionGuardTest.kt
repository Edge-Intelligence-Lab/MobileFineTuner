package com.mobilefinetuner.visualizer.continuity

import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Test

class CpuContentionGuardTest {
    private val guard = CpuContentionGuard()
    private var time = 0L
    private var runNs = 0L
    private var waitNs = 0L

    private fun window(runMs: Int, waitMs: Int, wallMs: Long = 1000): String? {
        time += wallMs
        runNs += runMs * 1000000L
        waitNs += waitMs * 1000000L
        return guard.observe(time, SchedulerTimes(runNs, waitNs))
    }

    @Test fun utilizationSleepingAndBurstsAreNotContention() {
        repeat(6) { assertNull(window(995, 5)) }
        repeat(6) { assertNull(window(10, 90)) }
        repeat(3) {
            assertNull(window(250, 650))
            assertNull(window(250, 650))
            assertNull(window(800, 100))
        }
    }

    @Test fun requiresThreeSecondsAndUsesLowerContinuationThreshold() {
        assertNull(window(0, 0))
        repeat(5) { assertNull(window(450, 300)) } // 40% cannot arm
        assertNull(window(450, 450))
        assertNull(window(450, 300))
        assertNotNull(window(450, 300))
    }

    @Test fun missingRegressedOrStaleSamplesDiscardHistory() {
        assertNull(window(0, 0))
        repeat(2) { assertNull(window(300, 600)) }
        assertNull(guard.observe(time, null))
        repeat(3) { assertNull(window(300, 600)) }
        assertNull(window(300, 600, 3000))
        repeat(2) { assertNull(window(300, 600)) }
        runNs = 0L
        assertNull(window(300, 600))
        repeat(2) { assertNull(window(300, 600)) }
        assertNotNull(window(300, 600))
    }
}
