package com.mobilefinetuner.visualizer.continuity

import android.app.ActivityManager
import android.content.Context
import android.content.Intent
import android.os.*
import android.util.Log
import androidx.compose.ui.test.hasText
import androidx.compose.ui.test.isEnabled
import androidx.compose.ui.test.junit4.createEmptyComposeRule
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.performClick
import androidx.compose.ui.test.performScrollTo
import androidx.test.core.app.ActivityScenario
import androidx.test.core.app.ApplicationProvider
import androidx.test.platform.app.InstrumentationRegistry
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Assume.assumeTrue
import org.junit.Rule
import org.junit.Test
import java.io.Closeable
import java.io.File
import java.util.concurrent.FutureTask
import java.util.concurrent.LinkedBlockingQueue
import java.util.concurrent.TimeUnit

class PressureContinuityTest {
    @get:Rule val ui = createEmptyComposeRule()
    private val instrumentation = InstrumentationRegistry.getInstrumentation()
    private val context = ApplicationProvider.getApplicationContext<Context>()
    private val store = TaskStore(context)
    private val runner = TrainingRunner(context)
    private val steps = 256
    private val tag = "MftPressureE2E"
    private val mib = 1048576L

    private fun memory() = ActivityManager.MemoryInfo().also {
        context.getSystemService(ActivityManager::class.java).getMemoryInfo(it)
    }

    private fun healthy() = memory().also {
        assumeTrue("Requires >=4096 MiB currently available to the emulator",
            !it.lowMemory && it.availMem / mib >= 4096)
    }

    private fun until(timeoutMs: Long, reason: String, pollMs: Long = 100, predicate: () -> Boolean) {
        val end = SystemClock.elapsedRealtime() + timeoutMs
        while (!predicate()) {
            check(SystemClock.elapsedRealtime() < end) { "Timed out: $reason" }
            SystemClock.sleep(pollMs)
        }
    }

    private inner class Pressure(scenario: ActivityScenario<TrainingActivity>) : Closeable {
        private val replies = LinkedBlockingQueue<Message>()
        private val client = Messenger(Handler(Looper.getMainLooper()) {
            replies.add(Message.obtain(it))
            true
        })
        private val remote: Messenger
        private val initialDirty: Int
        var held = 0
            private set
        var dirtyDelta = 0
            private set

        private fun reply(type: Int): Message {
            val message = checkNotNull(replies.poll(10, TimeUnit.SECONDS)) { "No pressure reply for $type" }
            check(message.what == type) { "Pressure command $type refused: event=${message.what}, held=${message.arg1}" }
            return message
        }

        init {
            scenario.onActivity { activity ->
                activity.startActivity(Intent().setClassName(instrumentation.context.packageName,
                    PressureActivity::class.java.name).putExtra("client", client))
            }
            val ready = reply(PressureActivity.READY)
            check(ready.arg1 != android.os.Process.myPid())
            remote = ready.replyTo
            initialDirty = ready.arg2
            Log.i(tag, "companion_pid=${ready.arg1} initial_private_dirty_kib=$initialDirty")
        }

        fun grow() {
            remote.send(Message.obtain(null, PressureActivity.GROW))
            val ack = reply(PressureActivity.GROW)
            check(ack.arg1 == held + PressureActivity.CHUNK_MB)
            held = ack.arg1
            dirtyDelta = ack.arg2 - initialDirty
        }

        override fun close() {
            remote.send(Message.obtain(null, PressureActivity.RELEASE))
            val ack = reply(PressureActivity.RELEASE)
            Log.i(tag, "released_mib=$held private_dirty_after_release_kib=${ack.arg2}")
        }
    }

    private fun sessions(id: String) = File(store.task(id), "run/sessions").listFiles().orEmpty()
    private fun checkpointDir(id: String) = File(store.task(id),
        "run/training_state/" + store.status(id).getString("checkpoint_generation"))

    private fun assertHeld(id: String) {
        val status = store.status(id)
        val control = status.getJSONObject("control")
        assertEquals("memory_hold", status.getString("state"))
        assertFalse(status.getBoolean("execution_active"))
        assertFalse(control.getBoolean("enabled"))
        assertFalse(control.getBoolean("automatic"))
        assertTrue(control.getString("memory_hold_reason").contains("Memory pressure"))
        Log.i(tag, "hold generation=${control.getLong("resume_generation")} ${control.getString("memory_hold_reason")}")
    }

    private fun interrupt(id: String, pressure: Pressure, minimum: Int): Int {
        val gen = store.allow(id, true, false)
        val training = FutureTask { runner.run(id, true, expectedGeneration = gen) }
        Thread(training, "pressure-fixture-runner").start()
        try {
            var observed = 0
            until(15000, "first native update", pollMs = 5) {
                check(!training.isDone) { "Early exit: ${training.get()} ${store.status(id)}" }
                observed = store.checkpoint(id)?.optInt("committed_updates") ?: 0
                observed >= 1
            }
            val before = memory().availMem / mib
            check(before > minimum && store.control(id).getBoolean("enabled")) {
                "Already under pressure before stimulation: available=$before minimum=$minimum"
            }
            var avail: Long
            do {
                pressure.grow()
                avail = memory().availMem / mib
            } while (avail > minimum - 48 && pressure.held < PressureActivity.CAP_MB)
            Log.i(tag, "update_before_pressure=$observed before_mib=$before pressure_avail_mib=$avail held_mib=${pressure.held} " +
                "private_dirty_delta_kib=${pressure.dirtyDelta}")
            assertTrue("Cap reached without crossing real emergency floor", avail < minimum)
            assertTrue("Companion did not retain dirty pages", pressure.dirtyDelta >= pressure.held * 768)
            assertEquals(TrainingRunner.Outcome.SUSPENDED, training.get(25, TimeUnit.SECONDS))
            assertHeld(id)
            val checkpoint = store.status(id)
            assertEquals("suspended", checkpoint.getString("committed_status"))
            val stopped = checkpoint.getInt("committed_updates")
            assertTrue(stopped in observed until steps)
            val attempts = File(store.task(id), "attempts").list()!!.toSet()
            assertEquals(1, attempts.size)
            assertEquals(TrainingRunner.Outcome.DISABLED, runner.run(id, true))

            // Even a newly admitted generation may not launch a child while pressure is held.
            val newer = store.allow(id, true, false)
            assertTrue(newer > gen)
            assertEquals(TrainingRunner.Outcome.WAITING, runner.run(id, false, expectedGeneration = newer))
            assertHeld(id)
            assertEquals(attempts, File(store.task(id), "attempts").list()!!.toSet())
            assertEquals(1, sessions(id).size)
            assertEquals(checkpoint.getString("checkpoint_generation"),
                store.status(id).getString("checkpoint_generation"))
            Log.i(tag, "safely_suspended_at=$stopped newer_launch_blocked=true")
            return stopped
        } finally {
            if (!training.isDone) {
                store.pause(id)
                training.get(155, TimeUnit.SECONDS)
            }
        }
    }

    @Test fun realPressureThenManualResumeMatchesBaseline() = exercise(null)
    @Test fun realPressureThenRaggedResumeMatchesSameSchedule() = exercise(3)

    private fun exercise(cap: Int?) {
        assumeTrue("Opt in with -e pressureE2E true",
            InstrumentationRegistry.getArguments().getString("pressureE2E") == "true")
        assumeTrue("Requires an arm64 Android emulator", Build.HARDWARE in listOf("ranchu", "goldfish") &&
            "arm64-v8a" in Build.SUPPORTED_ABIS)
        healthy()
        assumeTrue("Requires idle trainer and healthy battery, thermal and disk conditions",
            !store.executionBusy() && Resources.read(context).nonMemoryBlock(false) == null)
        if (Build.VERSION.SDK_INT >= 33) instrumentation.uiAutomation.grantRuntimePermission(
            context.packageName, android.Manifest.permission.POST_NOTIFICATIONS)

        val tasks = mutableListOf<String>()
        fun task() = store.createDemo().also { id ->
            tasks.add(id)
            TaskStore.write(File(store.task(id), "spec.json"),
                store.spec(id).put("steps", steps).put("slice_seconds", 120).also {
                    if (cap != null) it.put("batch_size", 4)
                })
        }
        try {
            val baseline = task()
            store.allow(baseline, false, false)
            assertEquals(TrainingRunner.Outcome.COMPLETED, runner.run(baseline, false))
            val split = task()
            sessions(baseline).single().walkTopDown().single { it.name == "lora_initial.safetensors" }
                .copyTo(File(store.task(split), "initial.safetensors"))

            ActivityScenario.launch<TrainingActivity>(TrainingService.openTaskIntent(context, split)).use { scenario ->
                var entry: Long
                val stopped = Pressure(scenario).use { pressure ->
                    val ambient = healthy()
                    val minimum = (ambient.availMem / mib - 256).toInt()
                    entry = maxOf(minimum.toLong(), minOf(ambient.totalMem / mib / 8, 1024L),
                        ambient.threshold / mib + 128) + 128
                    assumeTrue("Insufficient admission margin", ambient.availMem / mib >= entry + 64)
                    TaskStore.write(File(store.task(split), "spec.json"), store.spec(split).put("min_available_mb", minimum))
                    Log.i(tag, "ambient_mib=${ambient.availMem / mib} total_mib=${ambient.totalMem / mib} " +
                        "threshold_mib=${ambient.threshold / mib} min_mib=$minimum entry_mib=$entry")
                    interrupt(split, pressure, minimum)
                }
                until(15000, "memory recovery to admission floor $entry") {
                    memory().let { !it.lowMemory && it.availMem / mib >= entry }
                }
                assertHeld(split)
                assertEquals(TrainingRunner.Outcome.DISABLED, runner.run(split, true))
                val heldGeneration = store.control(split).getLong("resume_generation")
                Log.i(tag, "recovered_mib=${memory().availMem / mib} automatic_still_disabled=true")
                ui.waitUntil(10000) {
                    ui.onAllNodes(hasText("Training stopped to free memory. When ready, tap Start / Resume."))
                        .fetchSemanticsNodes().isNotEmpty() &&
                        ui.onAllNodes(hasText("Start / Resume") and isEnabled()).fetchSemanticsNodes().isNotEmpty()
                }
                if (cap != null) {
                    ui.onNodeWithText("Keep previous batch (4)").performScrollTo().performClick()
                    ui.onNodeWithText("Use physical batch $cap").performClick()
                }
                ui.onNodeWithText("Start / Resume").performScrollTo().performClick()
                until(155000, "manual Resume completion") {
                    val status = store.status(split)
                    check(status.optString("state") != "failed") { status.toString() }
                    status.optString("committed_status") == "completed" && !status.getBoolean("execution_active")
                }
                assertEquals(2, sessions(split).size)
                val final = store.status(split)
                val resumedGeneration = final.getJSONObject("control").getLong("resume_generation")
                assertTrue(resumedGeneration > heldGeneration)
                assertFalse(final.getJSONObject("control").has("memory_hold_reason"))
                assertEquals(steps, final.getInt("committed_updates"))
                assertEquals(stopped, TrainingRunner.sessionStartUpdate(File(store.task(split), "run"),
                    final.getString("committing_session")))
                val reference = if (cap == null) baseline else task().also { id ->
                    sessions(baseline).single().walkTopDown().single { it.name == "lora_initial.safetensors" }
                        .copyTo(File(store.task(id), "initial.safetensors"))
                    store.allow(id, false, false)
                    assertEquals(TrainingRunner.Outcome.SUSPENDED, runner.run(id, false, stopAfter = stopped))
                    store.admitStart(id, false, false, resumeMicrobatch = cap)
                    assertEquals(TrainingRunner.Outcome.COMPLETED, runner.run(id, false))
                    assertEquals(cap, final.getInt("committed_physical_batch"))
                }
                val a = checkpointDir(reference)
                val b = checkpointDir(split)
                assertArrayEquals(File(a, "state.safetensors").readBytes(), File(b, "state.safetensors").readBytes())
                val ma = JSONObject(File(a, "state.json").readText())
                val mb = JSONObject(File(b, "state.json").readText())
                assertEquals(ma.getJSONObject("optimizer").toString(), mb.getJSONObject("optimizer").toString())
                assertEquals(ma.getString("dataset_state"), mb.getString("dataset_state"))
                Log.i(tag, "PASS baseline=$reference cap=$cap split=$split resumed_generation=$resumedGeneration " +
                    "resumed_from=$stopped updates=$steps exact_lora_optimizer=true")
            }
        } finally {
            for (id in tasks) {
                Scheduling.cancel(context, id, store.pause(id))
                until(155000, "trainer cleanup for $id") { !store.isExecuting(id) }
                store.task(id).deleteRecursively()
            }
        }
    }
}
