package com.mobilefinetuner.visualizer.continuity

import android.content.Context
import android.os.SystemClock
import android.provider.Settings
import android.util.AtomicFile
import org.json.JSONObject
import java.io.File
import java.io.RandomAccessFile

/** App-wide: the epoch also disarms already queued starts after the hour expires. */
internal class FocusWindow(private val context: Context) {
    internal data class Gate(val epoch: Long, val remainingMs: Long, val resumeAfterNs: Long, val sameBoot: Boolean) {
        fun permits(control: JSONObject) = remainingMs == 0L && control.optLong("focus_epoch") == epoch
    }
    private val root = context.filesDir
    private val state = File(root, "training_focus.json")
    private fun boot() = Settings.Global.getInt(context.contentResolver, Settings.Global.BOOT_COUNT)
    private fun read(): JSONObject = try {
        AtomicFile(state).openRead().bufferedReader().use { JSONObject(it.readText()) }
    } catch (_: java.io.FileNotFoundException) { JSONObject() }
    private fun <T> lock(block: () -> T): T = synchronized(monitor) {
        RandomAccessFile(File(root, "training_focus.lock"), "rw").use { f -> f.channel.lock().use { block() } }
    }
    fun <T> withGate(block: (Gate) -> T): T = lock {
        val s = read()
        val now = SystemClock.elapsedRealtime()
        val boot = boot()
        if (s.optLong("epoch") > 0 && s.optInt("boot", -1) != boot) {
            // Wall time is a best-effort bridge across reboot, never the running timer.
            val remaining = (s.optLong("until_wall") - System.currentTimeMillis()).coerceIn(0L, HOUR)
            s.put("boot", boot).put("until_elapsed", if (remaining > 0) now + remaining else 0L)
                .put("resume_after_ns", if (remaining > 0) (now + remaining) * 1000000L else 0L)
            if (remaining == 0L) s.put("until_wall", 0L)
            TaskStore.write(state, s)
        }
        val until = s.optLong("until_elapsed")
        val remaining = (until - now).coerceIn(0L, HOUR)
        if (remaining == 0L && until > 0L) {
            s.put("until_elapsed", 0L).put("until_wall", 0L)
            TaskStore.write(state, s)
        }
        block(Gate(s.optLong("epoch"), remaining, s.optLong("resume_after_ns"), s.optInt("boot", -1) == boot))
    }
    fun start() = lock {
        val until = SystemClock.elapsedRealtime() + HOUR
        TaskStore.write(state, JSONObject().put("epoch", read().optLong("epoch") + 1)
            .put("boot", boot()).put("resume_after_ns", until * 1000000L)
            .put("until_elapsed", until)
            .put("until_wall", System.currentTimeMillis() + HOUR))
    }
    fun end() = lock {
        TaskStore.write(state, read().put("boot", boot()).put("resume_after_ns", SystemClock.elapsedRealtimeNanos())
            .put("until_elapsed", 0L).put("until_wall", 0L))
    }
    companion object {
        private val monitor = Any()
        const val HOUR = 3_600_000L
    }
}
