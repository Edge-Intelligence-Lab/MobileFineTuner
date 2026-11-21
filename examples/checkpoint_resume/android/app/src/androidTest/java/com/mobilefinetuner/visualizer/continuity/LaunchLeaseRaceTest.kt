package com.mobilefinetuner.visualizer.continuity

import android.content.Context
import android.content.ContextWrapper
import androidx.test.core.app.ApplicationProvider
import org.junit.After
import org.junit.Assert.*
import org.junit.Test
import java.io.File
import java.util.UUID
import java.util.concurrent.CountDownLatch
import java.util.concurrent.FutureTask
import java.util.concurrent.TimeUnit

class LaunchLeaseRaceTest {
    private val app = ApplicationProvider.getApplicationContext<Context>()
    private val dir = File(app.filesDir, "launch-test-" + UUID.randomUUID()).apply { mkdirs() }
    private val context = object : ContextWrapper(app) { override fun getFilesDir() = dir }
    private val store = TaskStore(context)
    private val id = store.createDemo()

    @After fun cleanup() { dir.deleteRecursively() }

    private fun <T> async(block: () -> T): Pair<Thread, FutureTask<T>> {
        val future = FutureTask<T> { block() }
        return Thread(future).apply { start() } to future
    }

    // Wait for an actual contended monitor rather than assuming a sleep scheduled both threads.
    private fun awaitBlocked(thread: Thread) {
        val end = System.nanoTime() + TimeUnit.SECONDS.toNanos(5)
        while (thread.isAlive && thread.state != Thread.State.BLOCKED && System.nanoTime() < end) Thread.yield()
        assertEquals(Thread.State.BLOCKED, thread.state)
    }

    @Test fun deniedFinalCheckNeverOpensRunOrLaunches() {
        val old = store.allow(id, true, false)
        store.pause(id)
        fun denied(g: Long, background: Boolean = false, cancelled: Boolean = false) {
            assertNull(store.launchIfCurrent<Unit>(id, g, background, { cancelled }) { error("launched") })
            assertFalse(File(store.task(id), "run").exists())
        }
        denied(old)
        val next = store.allow(id, false, false)
        denied(old)
        denied(next, background = true)
        denied(next, cancelled = true)
        FocusWindow(context).start()
        denied(next)
        FocusWindow(context).end()
        denied(next) // no resurrection of the previous focus epoch
    }

    @Test fun pauseThatLosesLaunchWritesFreshMarker() {
        val generation = store.allow(id, false, false)
        val marker = File(store.task(id), "run/SUSPEND")
        marker.parentFile!!.mkdirs()
        marker.writeText("stale")
        val entered = CountDownLatch(1)
        val release = CountDownLatch(1)
        val launch = async {
            store.launchIfCurrent(id, generation, false, { false }) {
                assertFalse(marker.exists())
                entered.countDown()
                check(release.await(10, TimeUnit.SECONDS))
                true
            }
        }.second
        assertTrue(entered.await(5, TimeUnit.SECONDS))
        val (thread, pause) = async { TaskStore(context).pause(id) }
        try { awaitBlocked(thread) } finally { release.countDown() }
        assertEquals(true, launch.get(5, TimeUnit.SECONDS))
        pause.get(5, TimeUnit.SECONDS)
        assertTrue(marker.isFile)
        assertEquals("", marker.readText())
        assertFalse(store.control(id).getBoolean("enabled"))
    }

    @Test fun idleStatusProbeCannotRejectOwnerOrMakeStartFail() {
        val other = store.createDemo()
        val entered = CountDownLatch(1)
        val release = CountDownLatch(1)
        val read = async {
            store.readActiveExecution(id) { active ->
                entered.countDown()
                check(release.await(10, TimeUnit.SECONDS))
                active
            }
        }.second
        assertTrue(entered.await(5, TimeUnit.SECONDS))
        val (thread, start) = async {
            val contender = TaskStore(context)
            contender.withTrainingExecution(id, { error("status reader treated as owner") }) {
                assertTrue(store.isExecuting(id))
                assertFalse(store.anotherTaskExecuting(id))
                assertTrue(store.anotherTaskExecuting(other))
                assertTrue(store.withTrainingExecution(id, { it }) { error("duplicate owner") })
                assertFalse(store.withTrainingExecution(other, { it }) { error("second owner") })
                true
            }
        }
        try { awaitBlocked(thread) } finally { release.countDown() }
        assertFalse(read.get(5, TimeUnit.SECONDS))
        assertTrue(start.get(5, TimeUnit.SECONDS))
        assertFalse(store.isExecuting(id))
        assertFalse(store.executionBusy())
    }

    @Test fun pauseBetweenProcessStartAndNativeResumeIsNotConsumedAsStale() {
        val g = store.allow(id, false, false)
        val task = store.task(id)
        val run = File(task, "run")
        val binary = File(app.applicationInfo.nativeLibraryDir, "libmft_qwen.so").path
        fun launch(args: List<String>, tag: String): Process = checkNotNull(
            store.launchIfCurrent(id, g, false, { false }) {
                val pb = ProcessBuilder(args).directory(task).redirectErrorStream(true)
                    .redirectOutput(File(dir, "$tag.log"))
                pb.environment()["MFT_NUM_THREADS"] = "1"
                // This legacy handoff fixture needs writable stdin as an explicit pre-exec gate.
                pb.environment()["MFT_PARENT_MANAGES_SUSPEND"] = "1"
                pb.start()
            })
        fun finished(p: Process, tag: String) {
            assertTrue(p.waitFor(60, TimeUnit.SECONDS))
            assertEquals(File(dir, "$tag.log").readText().takeLast(1500), 0, p.exitValue())
        }
        val first = launch(listOf(binary, "--run_dir", run.path, "--model_dir", File(task, "model").path,
            "--jsonl_train", File(task, "train.jsonl").path, "--jsonl_valid", File(task, "valid.jsonl").path,
            "--batch_size", "1", "--seq_len", "16", "--steps", "3", "--lora_r", "2", "--lora_dropout", "0",
            "--suspend_after_updates", "1"), "initial")
        try { finished(first, "initial") } finally { if (first.isAlive) first.destroyForcibly().waitFor() }
        assertEquals(1, store.checkpoint(id)!!.getInt("committed_updates"))
        val marker = File(run, "SUSPEND").apply { createNewFile() }
        // No timer or scheduler assumptions: native cannot exec until the parent releases stdin.
        val child = launch(listOf("/system/bin/sh", "-c", "read gate || exit 2; exec \"\$@\"", "gate",
            binary, "--resume_state", run.path), "resume")
        try {
            assertFalse(marker.exists())
            store.pause(id)
            assertTrue(marker.isFile)
            child.outputStream.use { it.write("go\n".toByteArray()) }
            finished(child, "resume")
            assertEquals(1, store.checkpoint(id)!!.getInt("committed_updates"))
            assertEquals("suspended", store.checkpoint(id)!!.getString("committed_status"))
            assertTrue(marker.exists())
        } finally { if (child.isAlive) child.destroyForcibly().waitFor() }
    }
}
