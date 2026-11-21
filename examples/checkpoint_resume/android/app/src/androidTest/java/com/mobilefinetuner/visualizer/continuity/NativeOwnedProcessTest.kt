package com.mobilefinetuner.visualizer.continuity

import android.content.Context
import android.content.ContextWrapper
import android.system.Os
import android.system.OsConstants
import androidx.test.core.app.ApplicationProvider
import org.junit.After
import org.junit.Assert.*
import org.junit.Test
import java.io.File
import java.util.UUID
import java.util.concurrent.CountDownLatch
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit

class NativeOwnedProcessTest {
    private val app = ApplicationProvider.getApplicationContext<Context>()
    private val dir = File(app.filesDir, "jni-owner-" + UUID.randomUUID()).apply { mkdirs() }
    private val context = object : ContextWrapper(app) { override fun getFilesDir() = dir }
    private val pool = Executors.newFixedThreadPool(3)

    @After fun cleanup() {
        pool.shutdownNow()
        dir.deleteRecursively()
    }

    private fun shell(script: String, vararg args: String) = NativeOwnedProcess.prepare(
        listOf("/system/bin/sh", "-c", script, "test") + args, dir, File(dir, "attempt.log"), 2)()

    @Test fun naturalExitIsCachedAndExecUsesLiteralArgumentsAndManagedEnvironment() {
        val literal = "one value; not a command"
        val p = shell("""printf '%s\n' "${'$'}PWD" "${'$'}1" "${'$'}MFT_NUM_THREADS" "${'$'}MFT_PARENT_MANAGES_SUSPEND" >&2; exit 23""", literal)
        try {
            assertTrue(p.waitFor(5, TimeUnit.SECONDS))
            p.terminate()
            assertFalse(p.isAlive)
            assertEquals(23, p.exitValue())
            val lines = File(dir, "attempt.log").readLines()
            assertEquals(dir.canonicalFile, File(lines.first()).canonicalFile)
            assertEquals(listOf(literal, "2", "1"), lines.drop(1))
        } finally { p.terminate() }
    }

    @Test fun concurrentExitWaitAndReleaseObserveOneTerminalStatus() {
        // A second, independently owned child must survive every reap/release of its peers.
        val other = NativeOwnedProcess.prepare(listOf("/system/bin/sleep", "90"), dir, File(dir, "other"), 1)()
        try {
            repeat(24) {
                val p = shell("exit 23")
                try {
                    val start = CountDownLatch(1)
                    val waiter = pool.submit<Int> {
                        start.await()
                        check(p.waitFor(5, TimeUnit.SECONDS))
                        p.exitValue()
                    }
                    val killer = pool.submit<Int> { start.await(); p.terminate(); p.exitValue() }
                    start.countDown()
                    val status = waiter.get(6, TimeUnit.SECONDS)
                    assertTrue(status == 23 || status == 137)
                    assertEquals(status, killer.get(6, TimeUnit.SECONDS))
                    p.terminate()
                    assertTrue(other.isAlive)
                } finally { p.terminate() }
            }
        } finally { other.terminate() }
        assertEquals(137, other.exitValue())
    }

    @Test fun childBlockedBeforeExecDoesNotHoldFocusOrTaskGate() {
        val store = TaskStore(context)
        val id = store.createDemo()
        val generation = store.allow(id, false, false)
        val fifo = File(dir, "blocked-log")
        Os.mkfifo(fifo.path, 384)
        val reached = File(dir, "exec-ran")
        val spawn = NativeOwnedProcess.prepare(listOf("/system/bin/touch", reached.path), dir, fifo, 1)
        // No reader: child setup cannot finish opening stdout, let alone exec or report readiness.
        val launched = pool.submit<NativeOwnedProcess> {
            checkNotNull(store.launchIfCurrent(id, generation, false, { false }, spawn))
        }
        try {
            val child = launched.get(5, TimeUnit.SECONDS)
            try {
                pool.submit { FocusWindow(context).start(); store.pause(id) }.get(5, TimeUnit.SECONDS)
                assertFalse(child.waitFor(0, TimeUnit.SECONDS))
                assertFalse(reached.exists())
                pool.submit { child.terminate() }.get(5, TimeUnit.SECONDS)
                assertEquals(137, child.exitValue())
            } finally { child.terminate() }
        } finally {
            // Unstick even a regressed spawn implementation that waited for child setup.
            val fd = Os.open(fifo.path, OsConstants.O_RDWR or OsConstants.O_NONBLOCK, 0)
            try { launched.get(5, TimeUnit.SECONDS).terminate() } finally { Os.close(fd) }
        }
    }

    @Test fun stoppedChildCanBeReleasedWithoutAResumeSignal() {
        val child = shell("echo \$\$; kill -STOP \$\$; exit 19")
        try {
            val log = File(dir, "attempt.log")
            val end = System.nanoTime() + TimeUnit.SECONDS.toNanos(5)
            fun stopped(): Boolean {
                val pid = if (log.exists()) log.readText().trim().toIntOrNull() else null
                return pid != null && File("/proc/$pid/status").readLines().any {
                    it.startsWith("State:") && it.substringAfter(':').trim().startsWith("T")
                }
            }
            while (!stopped() && System.nanoTime() < end) Thread.sleep(5)
            assertTrue(stopped())
            pool.submit { child.terminate() }.get(5, TimeUnit.SECONDS)
            assertEquals(137, child.exitValue())
        } finally { child.terminate() }
    }
}
