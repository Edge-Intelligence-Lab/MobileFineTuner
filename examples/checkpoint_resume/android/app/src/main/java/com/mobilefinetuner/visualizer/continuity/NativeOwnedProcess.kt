package com.mobilefinetuner.visualizer.continuity

import android.os.SystemClock
import java.io.File
import java.io.IOException
import java.util.concurrent.TimeUnit

/** The runner is the sole waiter. No Java Process object/reaper is created for this child. */
internal class NativeOwnedProcess private constructor() {
    private var pid = 0
    private var status = RUNNING
    private external fun nativeSpawn(spec: ByteArray, env: ByteArray): Int
    private external fun nativeCollect(pid: Int, terminate: Boolean): Int

    @Synchronized private fun collect(terminate: Boolean = false): Int {
        if (pid != 0) {
            status = nativeCollect(pid, terminate)
            if (status != RUNNING) pid = 0 // Includes ECHILD: never reuse a lost identity.
        }
        if (status == LOST) throw IOException("Lost ownership of native trainer")
        return status
    }

    val isAlive get() = collect() == RUNNING
    fun exitValue() = collect().also { if (it == RUNNING) throw IllegalThreadStateException("Trainer still running") }
    fun terminate() { collect(terminate = true) }
    fun waitFor(timeout: Long, unit: TimeUnit): Boolean {
        val started = SystemClock.elapsedRealtimeNanos()
        val nanos = unit.toNanos(timeout.coerceAtLeast(0))
        while (isAlive) {
            val left = nanos - (SystemClock.elapsedRealtimeNanos() - started)
            if (left <= 0) return false
            TimeUnit.NANOSECONDS.sleep(minOf(left, 20_000_000L))
        }
        return true
    }

    companion object {
        private const val RUNNING = -1
        private const val LOST = -2
        init { System.loadLibrary("mft_owned_process") }

        private fun pack(fields: List<String>): ByteArray {
            require(fields.none { '\u0000' in it }) { "NUL in process specification" }
            return fields.joinToString("\u0000", postfix = "\u0000").toByteArray(Charsets.UTF_8)
        }

        // Load JNI and marshal UTF-8 (not JNI modified UTF-8) before entering the launch gate.
        fun prepare(argv: List<String>, cwd: File, log: File, threads: Int): () -> NativeOwnedProcess {
            require(argv.isNotEmpty() && File(argv.first()).isAbsolute && cwd.isAbsolute && log.isAbsolute)
            require(threads > 0)
            val env = System.getenv().toMutableMap()
            for (key in listOf("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                    "MKL_NUM_THREADS", "MFT_NUM_THREADS", "BLIS_NUM_THREADS")) env[key] = threads.toString()
            env["MFT_PARENT_MANAGES_SUSPEND"] = "1"
            val spec = pack(listOf(cwd.path, log.path) + argv)
            val environment = pack(env.map { (k, v) -> "$k=$v" })
            return { NativeOwnedProcess().apply { pid = nativeSpawn(spec, environment) } }
        }
    }
}
