package com.mobilefinetuner.visualizer.continuity

import org.junit.Assert.*
import org.junit.Test
import java.io.File

class ResourcePolicyTest {
    private val healthy = Resources(true, 80, 0, false, 2048, 4096)
    @Test fun permitsHealthy() { assertNull(healthy.nonMemoryBlock(true)) }
    @Test fun rejectsUnplugged() { assertNotNull(healthy.copy(charging = false).nonMemoryBlock(true)) }
    @Test fun allowsExplicitBatteryMode() { assertNull(healthy.copy(charging = false).nonMemoryBlock(false)) }
    @Test fun rejectsLowOrUnknownBattery() {
        assertNotNull(healthy.copy(battery = 19).nonMemoryBlock(false))
        assertNotNull(healthy.copy(battery = -1).nonMemoryBlock(false))
    }
    @Test fun rejectsThermal() { assertNotNull(healthy.copy(thermal = 3).nonMemoryBlock(true)) }
    @Test fun rejectsMemory() {
        assertNotNull(MemoryGuard(512).beforeStart(healthy.copy(lowMemory = true)))
        assertNotNull(MemoryGuard(512).beforeStart(healthy.copy(availableMb = 511)))
    }
    @Test fun rejectsDisk() { assertNotNull(healthy.copy(diskMb = 127).nonMemoryBlock(true)) }
    @Test fun lowMemoryAndHardFloorYieldImmediately() {
        val guard = MemoryGuard(512)
        assertNotNull(guard.duringTraining(healthy.copy(lowMemory = true), 0))
        assertNotNull(guard.duringTraining(healthy.copy(availableMb = 511), 0))
        assertNotNull(guard.duringTraining(healthy.copy(lowThresholdMb = 800, availableMb = 799), 0))
    }
    @Test fun softPressureMustPersistAndAdmissionHasMargin() {
        val guard = MemoryGuard(512)
        val low = healthy.copy(totalMb = 12288, availableMb = 1000)
        assertNull(guard.duringTraining(low, 0))
        assertNull(guard.duringTraining(low, 999))
        assertNull(guard.duringTraining(low.copy(availableMb = 1024), 1000))
        assertNull(guard.duringTraining(low, 1200))
        assertNotNull(guard.duringTraining(low, 2200))
        assertNotNull(guard.beforeStart(low.copy(availableMb = 1151)))
        assertNull(guard.beforeStart(low.copy(availableMb = 1152)))
    }
    @Test fun noProgressBudgetFailure() { assertTrue(TrainingRunner.exhaustedWithoutProgress(4, 4, 1500, 1, false)) }
    @Test fun progressIsNotFailure() { assertFalse(TrainingRunner.exhaustedWithoutProgress(4, 5, 1500, 1, false)) }
    @Test fun manualOrResourceStopIsNotBudgetFailure() { assertFalse(TrainingRunner.exhaustedWithoutProgress(4, 4, 1500, 1, true)) }
    @Test fun finiteGraceAndWakeBudgetsShareTheSameBound() {
        assertEquals(150000L, TrainingRunner.executionBudgetMs(120, 30))
        assertEquals(360000L, TrainingRunner.executionBudgetMs(120, 240))
        assertEquals(540000L, TrainingRunner.executionBudgetMs(120, 420))
        assertEquals(550000L, TrainingRunner.wakeBudgetMs(120, 420))
        assertEquals(31000L, TrainingRunner.executionBudgetMs(1, 30))
    }
    @Test fun rejectsUnboundedGraceAndInvalidSlice() {
        for ((slice, grace) in listOf(0 to 30, 121 to 30, 30 to 29, 30 to 421, 30 to Int.MAX_VALUE)) {
            assertTrue(runCatching { TrainingRunner.executionBudgetMs(slice, grace) }.exceptionOrNull() is IllegalArgumentException)
        }
    }
    @Test fun sameTaskHandoffWaitIsFiniteAndCoversOneOldSlice() {
        assertEquals(TrainingRunner.wakeBudgetMs(120, 420), TrainingService.HANDOFF_WAIT_MS)
        assertFalse(TrainingService.handoffExpired(1000L, 550999L))
        assertTrue(TrainingService.handoffExpired(1000L, 551000L))
        assertTrue(TrainingService.handoffExpired(1000L, 600000L))
    }
    @Test fun nativeErrorSummaryIsBoundedAndFullLogIsPreserved() {
        val file = File.createTempFile("mft-error-", ".log")
        try {
            val log = "progress\n".repeat(4000) + "FATAL: missing model/config.json\n" + "cleanup\n".repeat(100)
            file.writeText(log)
            val summary = TrainingRunner.errorTail(file)
            assertTrue(summary.contains("missing model/config.json"))
            assertTrue(summary.length <= 1200)
            assertEquals(log, file.readText())
            file.writeText("ERROR: " + "x".repeat(20000))
            assertTrue(TrainingRunner.errorTail(file).length <= 1200)
        } finally { file.delete() }
    }
}
