package com.mobilefinetuner.visualizer.continuity

import android.app.ActivityManager
import android.app.NotificationManager
import android.content.Context
import android.content.Intent
import androidx.core.content.ContextCompat
import androidx.compose.ui.test.junit4.createEmptyComposeRule
import androidx.compose.ui.test.onAllNodesWithText
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.performClick
import androidx.compose.ui.test.performScrollTo
import androidx.compose.ui.test.isToggleable
import androidx.compose.ui.test.isEnabled
import androidx.compose.ui.test.hasText
import androidx.compose.ui.test.assertIsOff
import androidx.test.core.app.ActivityScenario
import androidx.test.core.app.ApplicationProvider
import androidx.test.platform.app.InstrumentationRegistry
import androidx.work.WorkManager
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Rule
import org.junit.Test
import org.junit.Before
import java.io.File
import java.util.concurrent.TimeUnit

class UiContinuityTest {
    @get:Rule val ui = createEmptyComposeRule()
    private val context = ApplicationProvider.getApplicationContext<Context>()
    private val store = TaskStore(context)
    @Before fun allowNotificationsForUiTests() {
        if (android.os.Build.VERSION.SDK_INT >= 33) {
            InstrumentationRegistry.getInstrumentation().uiAutomation
                .grantRuntimePermission(context.packageName, android.Manifest.permission.POST_NOTIFICATIONS)
        }
    }
    private fun await(message: String, seconds: Int = 90, check: () -> Boolean) {
        val end = android.os.SystemClock.elapsedRealtime() + seconds * 1000L
        while (android.os.SystemClock.elapsedRealtime() < end) {
            if (check()) return
            Thread.sleep(100)
        }
        fail(message)
    }
    private fun free(): Boolean = !store.executionBusy()
    private fun create(slice: Int): String {
        val id = store.createDemo()
        TaskStore.write(File(store.task(id), "spec.json"), store.spec(id).put("steps", 5000).put("slice_seconds", slice))
        store.allow(id, true, false)
        return id
    }
    private fun startUi(id: String) {
        ui.onAllNodesWithText("Task:", substring = true)[0].performScrollTo().performClick()
        ui.onNodeWithText(id).performScrollTo().performClick()
        ui.waitUntil(10000) {
            ui.onAllNodes(hasText("Start / Resume") and isEnabled()).fetchSemanticsNodes().isNotEmpty()
        }
        ui.onNodeWithText("Start / Resume").performScrollTo().performClick()
    }
    private fun servicePid(): Int? = context.getSystemService(ActivityManager::class.java).runningAppProcesses
        ?.firstOrNull { it.processName == context.packageName + ":training" }?.pid
    private fun sessions(id: String) = File(store.task(id), "run/sessions").listFiles()?.toList() ?: emptyList()
    @Test fun smallerPhysicalBatchSurvivesPressureHoldAndUiRelaunch() {
        val id = store.createDemo()
        val spec = store.spec(id).put("batch_size", 4).put("steps", 8)
        val file = File(store.task(id), "spec.json")
        TaskStore.write(file, spec)
        val generation = store.allow(id, false, false)
        try {
            assertEquals(TrainingRunner.Outcome.SUSPENDED, TrainingRunner(context).run(id, false, stopAfter = 1))
            assertEquals(4, store.checkpoint(id)!!.getInt("committed_physical_batch"))
            TaskStore.write(file, JSONObject(spec.toString()).put("min_available_mb", 1000000))
            assertTrue(store.holdForMemory(id, generation, "test pressure"))
            val intent = TrainingService.openTaskIntent(context, id)
            ActivityScenario.launch<TrainingActivity>(intent).use {
                ui.waitUntil(10000) { ui.onAllNodesWithText("Keep previous batch (4)").fetchSemanticsNodes().isNotEmpty() }
                ui.onNodeWithText("Keep previous batch (4)").performScrollTo().performClick()
                ui.onNodeWithText("Use physical batch 3").performClick()
                ui.onNodeWithText("Start / Resume").performScrollTo().performClick()
                await("smaller batch bypassed the pressure gate") {
                    val ctl = store.control(id)
                    ctl.optLong("resume_generation") > generation && ctl.has("memory_hold_reason") &&
                        !ctl.optBoolean("enabled") && free()
                }
                assertEquals(1, sessions(id).size)
                assertEquals(3, TaskStore(context).control(id).getInt("resume_microbatch"))
                assertEquals(4, store.checkpoint(id)!!.getInt("committed_physical_batch"))
            }
            TaskStore.write(file, spec)
            ActivityScenario.launch<TrainingActivity>(intent).use {
                ui.waitUntil(10000) { ui.onAllNodesWithText("Keep previous batch (3)").fetchSemanticsNodes().isNotEmpty() }
                ui.onNodeWithText("Start / Resume").performScrollTo().performClick()
                await("service did not commit the selected physical batch") {
                    store.checkpoint(id)?.optString("committed_status") == "completed" && free()
                }
                val checkpoint = store.checkpoint(id)!!
                assertEquals(3, checkpoint.getInt("committed_physical_batch"))
                assertEquals(8, checkpoint.getInt("committed_updates"))
                assertEquals(spec.toString(), store.spec(id).toString())
            }
        } finally { clean(id) }
    }

    @Test fun invalidPhysicalBatchRequestsLeaveAdmissionUntouched() {
        val id = store.createDemo()
        TaskStore.write(File(store.task(id), "spec.json"), store.spec(id).put("batch_size", 4))
        fun rejected(size: Int) {
            val before = store.control(id).toString()
            val error = runCatching { store.admitStart(id, true, false, resumeMicrobatch = size) }.exceptionOrNull()
            assertTrue("Expected batch $size to be rejected, got $error", error is IllegalArgumentException)
            assertEquals(before, store.control(id).toString())
        }
        rejected(3) // A valid cap still cannot be used for initialization.
        assertFalse(store.hasAttemptArtifacts(id))
        store.allow(id, false, false)
        assertEquals(TrainingRunner.Outcome.SUSPENDED, TrainingRunner(context).run(id, false, stopAfter = 1))
        store.pause(id)
        for (size in listOf(-1, 0, 5)) rejected(size)
        assertEquals(1, sessions(id).size)
    }

    @Test fun rotatingDuringManualAdmissionStillStartsOnce() {
        val id = store.createDemo()
        TaskStore.write(File(store.task(id), "spec.json"), store.spec(id).put("steps", 10))
        ActivityScenario.launch<TrainingActivity>(TrainingService.openTaskIntent(context, id)).use { scenario ->
            ui.waitUntil(10000) { ui.onAllNodes(hasText("Start / Resume") and isEnabled()).fetchSemanticsNodes().isNotEmpty() }
            ui.onNodeWithText("Start / Resume").performScrollTo()
            val held = java.util.concurrent.CountDownLatch(1)
            val release = java.util.concurrent.CountDownLatch(1)
            val lockOwner = Thread {
                store.updateControl(id) {
                    held.countDown()
                    check(release.await(30, TimeUnit.SECONDS))
                }
            }.apply { start() }
            try {
                assertTrue(held.await(5, TimeUnit.SECONDS))
                ui.onNodeWithText("Start / Resume").performClick()
                scenario.recreate()
            } finally { release.countDown(); lockOwner.join(5000) }
            await("accepted manual start lost after recreation") {
                store.status(id).optString("committed_status") == "completed" && free()
            }
            assertEquals(1, sessions(id).size)
            assertEquals(1L, store.control(id).getLong("resume_generation"))
        }
    }
    private fun nativePid(id: String) = JSONObject(sessions(id).single().resolve("events.jsonl").readLines().first()).getInt("pid")
    private fun parent(pid: Int) = File("/proc/$pid/stat").readText().substringAfterLast(") ").split(' ')[1].toInt()
    private fun clean(id: String) {
        Scheduling.cancel(context, id, store.pause(id))
        await("child still active after cleanup") { free() }
        Thread.sleep(600)
    }
    private fun nativeProcesses(id: String): List<Int> = File("/proc").listFiles().orEmpty().mapNotNull { entry ->
        val pid = entry.name.toIntOrNull() ?: return@mapNotNull null
        val command = runCatching { File(entry, "cmdline").readText() }.getOrNull() ?: return@mapNotNull null
        if (command.contains("libmft_qwen.so") && command.contains(store.task(id).absolutePath)) pid else null
    }
    private fun awaitSingleChild(id: String, message: String, check: () -> Boolean) = await(message) {
        assertTrue("Overlapping native children for $id", nativeProcesses(id).size <= 1)
        check()
    }
    private fun signal(pid: Int?, value: Int) {
        if (pid == null) return
        try { android.system.Os.kill(pid, value) }
        catch (e: android.system.ErrnoException) { if (e.errno != android.system.OsConstants.ESRCH) throw e }
    }
    private fun startManualService(id: String, generation: Long) {
        ContextCompat.startForegroundService(context, Intent(context, TrainingService::class.java)
            .putExtra("task", id).putExtra("resume_generation", generation))
    }
    private fun handoffRecorded(id: String, generation: Long): Boolean = runCatching {
        val state = JSONObject(File(store.task(id), "ui_state.json").readText())
        state.optString("state") == "handoff" && state.optLong("resume_generation") == generation
    }.getOrDefault(false)
    @Suppress("DEPRECATION")
    private fun serviceStopped(): Boolean = context.getSystemService(ActivityManager::class.java)
        .getRunningServices(Int.MAX_VALUE).none { it.service.className == TrainingService::class.java.name }
    private fun assertManualHandoff(id: String, oldSession: String, oldPid: Int, generation: Long, before: Int) {
        awaitSingleChild(id, "manual Resume never started a new native session") {
            val recordedPid = runCatching {
                JSONObject(sessions(id).single { it.name != oldSession }
                    .resolve("events.jsonl").readLines().first()).getInt("pid")
            }.getOrNull()
            recordedPid != null && recordedPid != oldPid && nativeProcesses(id) == listOf(recordedPid) &&
                store.status(id).optInt("committed_updates") > before
        }
        val next = sessions(id).single { it.name != oldSession }
        val pid = JSONObject(next.resolve("events.jsonl").readLines().first()).getInt("pid")
        assertNotEquals(oldPid, pid)
        assertFalse("old child still exists after handoff", File("/proc/$oldPid").exists())
        assertEquals(listOf(pid), nativeProcesses(id))
        assertEquals(checkNotNull(servicePid()), parent(pid))
        assertEquals(generation, store.control(id).getLong("resume_generation"))
        assertFalse(store.control(id).getBoolean("automatic"))
        assertTrue(store.control(id).getBoolean("enabled"))
        assertEquals(generation, store.status(id).getLong("resume_generation"))
        assertTrue(WorkManager.getInstance(context).getWorkInfosForUniqueWork("mft-$id")
            .get(10, TimeUnit.SECONDS).all { it.state.isFinished })
    }
    @Test fun actualUiHandsOffToWorker() {
        val id = create(1)
        try {
            ActivityScenario.launch(TrainingActivity::class.java).use {
                startUi(id)
                await("foreground never committed") { store.status(id).optInt("committed_updates") >= 2 }
                val first = sessions(id).single().name
                await("no background handoff") { sessions(id).any { it.name != first } && store.status(id).optInt("committed_updates") > 10 }
                assertTrue(store.control(id).getBoolean("automatic"))
                TaskStore.write(File(store.task(id), "ui_handoff_result.json"), JSONObject().put("passed", true).put("sessions", sessions(id).size))
            }
        } finally { clean(id) }
    }
    @Test fun actualUiSurvivesForegroundProcessDeathWithoutResumeClick() {
        val id = create(30)
        try {
            ActivityScenario.launch(TrainingActivity::class.java).use {
                startUi(id)
                await("foreground never committed") { store.status(id).optInt("committed_updates") >= 2 }
                val child = nativePid(id)
                val service = servicePid() ?: error("Missing foreground service")
                assertEquals(service, parent(child))
                val before = store.status(id).getInt("committed_updates")
                android.os.Process.killProcess(service)
                await("orphan after service death") { !File("/proc/$child").exists() }
                // No store.allow, Runner call or UI Resume after the kill.
                await("durable worker did not recover") { sessions(id).size >= 2 && store.status(id).optInt("committed_updates") > before + 2 }
                assertTrue(store.control(id).getBoolean("automatic"))
                TaskStore.write(File(store.task(id), "ui_death_result.json"), JSONObject().put("passed", true)
                    .put("before_updates", before).put("after_updates", store.status(id).getInt("committed_updates")))
            }
        } finally { clean(id) }
    }
    @Test fun actualNotificationPauseDisablesAutomaticContinuation() {
        val id = create(30)
        try {
            ActivityScenario.launch(TrainingActivity::class.java).use {
                startUi(id)
                await("foreground never committed") { store.status(id).optInt("committed_updates") >= 2 }
                val notification = context.getSystemService(NotificationManager::class.java).activeNotifications.single { it.id == 41 }.notification
                notification.actions.single { it.title.toString() == "Pause" }.actionIntent.send()
                await("notification failed to suspend") { !store.control(id).optBoolean("enabled") && free() }
                val before = store.status(id).getInt("committed_updates")
                Thread.sleep(12000) // Pending WorkManager entry must observe durable disabled state.
                assertFalse(store.control(id).getBoolean("automatic"))
                assertEquals(before, store.status(id).getInt("committed_updates"))
                TaskStore.write(File(store.task(id), "notification_pause_result.json"), JSONObject().put("passed", true).put("updates", before))
            }
        } finally { clean(id) }
    }
    @Test fun actualUiReplacesChargingConstraintBothDirections() {
        require(android.os.Build.HARDWARE in listOf("ranchu", "goldfish")) { "Battery override test is emulator-only" }
        val automation = InstrumentationRegistry.getInstrumentation().uiAutomation
        fun shell(command: String) = automation.executeShellCommand(command).use { descriptor ->
            android.os.ParcelFileDescriptor.AutoCloseInputStream(descriptor).use { it.readBytes() }
        }
        val id = create(1)
        store.allow(id, true, true)
        fun pending() = WorkManager.getInstance(context).getWorkInfosForUniqueWork("mft-$id").get()
            .filter { !it.state.isFinished }
        try {
            shell("dumpsys battery unplug"); shell("dumpsys battery set status 3"); shell("dumpsys battery set level 80")
            await("emulator did not enter unplugged state") { !Resources.read(context).charging }
            ActivityScenario.launch(TrainingActivity::class.java).use {
                startUi(id)
                await("charging-only task not enqueued") { pending().size == 1 }
                assertTrue(pending().single().constraints.requiresCharging())
                await("foreground should wait for charging") { store.status(id).optString("state") == "waiting" && free() }
                assertEquals(0, store.status(id).optInt("committed_updates"))
                ui.onAllNodes(isToggleable())[1].performScrollTo().performClick()
                ui.onNodeWithText("Start / Resume").performScrollTo().performClick()
                await("old charging constraint retained") { pending().size == 1 && !pending().single().constraints.requiresCharging() }
                await("no battery-powered background handoff") { sessions(id).size >= 2 && store.status(id).optInt("committed_updates") > 10 }
                // Switching back must replace the pending job and stop battery-powered execution.
                ui.onAllNodes(isToggleable())[1].performScrollTo().performClick()
                ui.onNodeWithText("Start / Resume").performScrollTo().performClick()
                await("charging-only constraint not restored") { pending().size == 1 && pending().single().constraints.requiresCharging() && free() }
                val stopped = store.status(id).getInt("committed_updates")
                Thread.sleep(3000)
                assertEquals(stopped, store.status(id).getInt("committed_updates"))
                TaskStore.write(File(store.task(id), "charging_switch_result.json"), JSONObject()
                    .put("passed", true).put("background_sessions", sessions(id).size).put("stopped_updates", stopped))
            }
        } finally {
            try { clean(id) } finally { shell("dumpsys battery reset") }
        }
    }
    @Test fun notificationBodyRoutesToItsTaskAndReusesActivity() {
        val a = create(30)
        val b = create(30)
        try {
            ActivityScenario.launch<TrainingActivity>(TrainingService.openTaskIntent(context, a)).use { scenario ->
                var original: TrainingActivity? = null
                scenario.onActivity { original = it }
                startUi(a)
                await("A did not start") { store.status(a).optInt("committed_updates") >= 2 }
                ui.onAllNodesWithText("Task:", substring = true)[0].performScrollTo().performClick()
                ui.onNodeWithText(b).performScrollTo().performClick()
                scenario.recreate()
                ui.waitUntil(10000) { ui.onAllNodesWithText("Task: $b").fetchSemanticsNodes(atLeastOneRootRequired = false).size == 1 }
                scenario.onActivity { original = it }
                val notification = context.getSystemService(NotificationManager::class.java)
                    .activeNotifications.single { it.id == 41 }.notification
                notification.contentIntent.send()
                ui.waitUntil(10000) { ui.onAllNodesWithText("Task: $a").fetchSemanticsNodes(atLeastOneRootRequired = false).size == 1 }
                scenario.onActivity {
                    assertSame(original, it)
                    assertEquals(a, it.intent.getStringExtra("task"))
                }
                assertTrue(sessions(b).isEmpty())
                notification.actions.single { it.title.toString() == "Pause" }.actionIntent.send()
                await("pause not persisted") { !store.control(a).optBoolean("automatic") && free() }
                ui.onAllNodes(isToggleable())[0].performScrollTo()
                ui.waitUntil(10000) {
                    runCatching { ui.onAllNodes(isToggleable())[0].assertIsOff() }.isSuccess
                }
            }
        } finally { clean(a); clean(b) }
    }
    @Test fun busyManualServiceRequestIsRejectedNotQueued() {
        val a = create(30)
        val b = create(30)
        try {
            ActivityScenario.launch<TrainingActivity>(TrainingService.openTaskIntent(context, a)).use {
                startUi(a)
                await("A did not start") { store.status(a).optInt("committed_updates") >= 2 }
                store.allow(b, false, false)
                ContextCompat.startForegroundService(context, Intent(context, TrainingService::class.java).putExtra("task", b))
                await("B was not rejected") { store.status(b).optString("state") == "rejected" }
                assertFalse(store.control(b).getBoolean("enabled"))
                assertFalse(store.control(b).getBoolean("automatic"))
                assertTrue(store.status(b).getString("detail").contains("not queued"))
                assertTrue(sessions(b).isEmpty())
                assertTrue(store.status(a).getBoolean("execution_active"))
            }
        } finally { clean(a); clean(b) }
    }
    @Test fun missingIntentTaskDoesNotSilentlySelectAnotherTask() {
        val id = store.createDemo()
        ActivityScenario.launch<TrainingActivity>(TrainingService.openTaskIntent(context, "missing-${System.nanoTime()}")).use {
            ui.waitUntil(10000) { ui.onAllNodesWithText("Requested task is unavailable; select an existing task.").fetchSemanticsNodes().isNotEmpty() }
            ui.onNodeWithText("Task: none").performScrollTo().assertExists()
            assertFalse(store.control(id).optBoolean("enabled"))
        }
    }
    @Test fun failedTaskOffersExplicitNewTaskWithoutDeletingFailure() {
        val id = store.createDemo()
        val log = File(store.task(id), "attempts/failed.log").apply { parentFile!!.mkdirs(); writeText("loading failed") }
        store.fail(id, "loading failed")
        val before = store.list().toSet()
        ActivityScenario.launch<TrainingActivity>(TrainingService.openTaskIntent(context, id)).use {
            ui.waitUntil(10000) { ui.onAllNodesWithText("Retry as new task (reinitialize)").fetchSemanticsNodes().isNotEmpty() }
            ui.onNodeWithText("Retry as new task (reinitialize)").performScrollTo().performClick()
            await("retry task not created") { store.list().toSet().minus(before).size == 1 }
            val next = store.list().toSet().minus(before).single()
            ui.waitUntil(10000) { ui.onAllNodesWithText("Task: $next").fetchSemanticsNodes().isNotEmpty() }
            assertEquals("loading failed", log.readText())
            assertFalse(store.control(next).getBoolean("enabled"))
            assertFalse(File(store.task(next), "run").exists())
        }
    }
    @Test fun pristineFailedTaskHasEnabledStartWithoutNewTaskCopy() {
        val id = store.createDemo()
        val generation = store.allow(id, false, false)
        store.fail(id, "scheduling admission failed", generation)
        val before = store.list().toSet()
        ActivityScenario.launch<TrainingActivity>(TrainingService.openTaskIntent(context, id)).use {
            ui.waitUntil(10000) {
                ui.onAllNodes(hasText("Start / Resume") and isEnabled()).fetchSemanticsNodes().isNotEmpty()
            }
            assertTrue(ui.onAllNodesWithText("Retry as new task (reinitialize)").fetchSemanticsNodes().isEmpty())
            assertEquals(before, store.list().toSet())
            assertFalse(store.hasAttemptArtifacts(id))
        }
    }
    @Test fun manualResumeHandsOffFromWorkerWithoutAutomaticFallback() {
        val id = create(30)
        var oldPid: Int? = null
        try {
            ActivityScenario.launch(TrainingActivity::class.java).use {
                await("previous service still running") { serviceStopped() }
                Scheduling.schedule(context, id, store.control(id).getLong("resume_generation"), replace = true)
                awaitSingleChild(id, "Worker never acquired the native lease") {
                    store.status(id).optInt("committed_updates") >= 2 && nativeProcesses(id).size == 1
                }
                val oldSession = sessions(id).single().name
                val pid = nativePid(id)
                oldPid = pid
                assertEquals(android.os.Process.myPid(), parent(pid))
                val oldGeneration = store.control(id).getLong("resume_generation")
                // Hold the real old child at the boundary so admission must exercise BUSY, not a lucky idle gap.
                signal(pid, android.system.OsConstants.SIGSTOP)
                val before = store.status(id).getInt("committed_updates")
                val next = store.allow(id, false, false)
                store.note(id, "starting", "manual Resume while Worker owns lease", next)
                startManualService(id, next)
                awaitSingleChild(id, "new Service did not retain the same-task BUSY request") { handoffRecorded(id, next) }
                assertTrue(store.isExecuting(id))
                assertEquals(listOf(pid), nativeProcesses(id))
                assertFalse(store.note(id, "suspended", "stale Worker result", oldGeneration))
                assertTrue(handoffRecorded(id, next))
                // Cancel after observing BUSY to make overlap deterministic; automatic is already false.
                Scheduling.cancel(context, id, next)
                signal(pid, android.system.OsConstants.SIGCONT)
                assertManualHandoff(id, oldSession, pid, next, before)
            }
        } finally {
            signal(oldPid, android.system.OsConstants.SIGCONT)
            clean(id)
        }
    }
    @Test fun manualResumeHandsOffThroughAlreadyActiveSameTaskService() {
        val id = create(30)
        var oldPid: Int? = null
        try {
            ActivityScenario.launch(TrainingActivity::class.java).use {
                val oldGeneration = store.allow(id, false, false)
                startManualService(id, oldGeneration)
                awaitSingleChild(id, "Service never acquired the native lease") {
                    store.status(id).optInt("committed_updates") >= 2 && nativeProcesses(id).size == 1
                }
                val oldSession = sessions(id).single().name
                val pid = nativePid(id)
                oldPid = pid
                assertEquals(checkNotNull(servicePid()), parent(pid))
                signal(pid, android.system.OsConstants.SIGSTOP)
                val before = store.status(id).getInt("committed_updates")
                val next = store.allow(id, false, false)
                startManualService(id, next)
                awaitSingleChild(id, "active Service dropped the newer same-task request") { handoffRecorded(id, next) }
                assertTrue(store.isExecuting(id))
                assertEquals(listOf(pid), nativeProcesses(id))
                assertFalse(store.note(id, "suspended", "stale Service result", oldGeneration))
                assertTrue(handoffRecorded(id, next))
                signal(pid, android.system.OsConstants.SIGCONT)
                assertManualHandoff(id, oldSession, pid, next, before)
            }
        } finally {
            signal(oldPid, android.system.OsConstants.SIGCONT)
            clean(id)
        }
    }
    @Test fun userPauseCancelsPendingSameTaskHandoff() {
        val id = create(30)
        var oldPid: Int? = null
        try {
            ActivityScenario.launch(TrainingActivity::class.java).use {
                startManualService(id, store.allow(id, false, false))
                awaitSingleChild(id, "Service never started before pending Pause") {
                    store.status(id).optInt("committed_updates") >= 2 && nativeProcesses(id).size == 1
                }
                val pid = nativePid(id)
                oldPid = pid
                signal(pid, android.system.OsConstants.SIGSTOP)
                val next = store.allow(id, false, false)
                startManualService(id, next)
                awaitSingleChild(id, "handoff was not pending before Pause") { handoffRecorded(id, next) }
                assertEquals(listOf(pid), nativeProcesses(id))
                store.pause(id)
                signal(pid, android.system.OsConstants.SIGCONT)
                awaitSingleChild(id, "Pause did not cancel the pending Resume") { free() && serviceStopped() }
                assertEquals(1, sessions(id).size)
                assertTrue(nativeProcesses(id).isEmpty())
                assertFalse(store.control(id).getBoolean("enabled"))
                assertFalse(store.control(id).getBoolean("automatic"))
                assertEquals(next, store.control(id).getLong("resume_generation"))
            }
        } finally {
            signal(oldPid, android.system.OsConstants.SIGCONT)
            clean(id)
        }
    }
}
