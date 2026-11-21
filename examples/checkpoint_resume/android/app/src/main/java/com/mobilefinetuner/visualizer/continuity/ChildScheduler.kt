package com.mobilefinetuner.visualizer.continuity

import android.os.Process
import android.system.Os
import org.json.JSONObject
import java.io.File

/** Created immediately before launch, under the execution lease. Never enumerates system PIDs. */
internal class ChildScheduler(run: File, private val threads: Int) {
    private val sessions = File(run, "sessions")
    private val previous = try {
        if (!sessions.exists()) emptySet() else sessions.list()?.toSet()
    } catch (_: SecurityException) { null }
    private var pid: Int? = null

    fun read(): SchedulerTimes? {
        try {
            val known = previous ?: return null
            val id = pid ?: run {
                val session = sessions.list()?.singleOrNull { it !in known } ?: return null
                val start = File(sessions, "$session/events.jsonl").bufferedReader().use { it.readLine() }
                    ?: return null
                val event = JSONObject(start)
                if (event.getString("type") != "session_start" || event.getString("session_id") != session) return null
                event.getInt("pid").takeIf { it > 0 } ?: return null
            }
            val proc = File("/proc/$id")
            if (Os.stat(proc.path).st_uid != Process.myUid()) return null
            val status = File(proc, "status").readLines()
            if (field(status, "PPid") != Process.myPid().toString()) return null
            // A configured pool larger than its affinity mask can compete with itself.
            val allowed = field(status, "Cpus_allowed_list") ?: return null
            if (allowed.split(',').sumOf { range ->
                val bounds = range.split('-')
                bounds.last().toInt() - bounds.first().toInt() + 1
            } < threads) return null
            val fields = File(proc, "schedstat").readText().trim().split(whitespace)
            if (fields.size < 3) return null
            val runNs = fields[0].toLong()
            val waitNs = fields[1].toLong()
            if (runNs < 0 || waitNs < 0) return null
            pid = id
            return SchedulerTimes(runNs, waitNs)
        } catch (_: Exception) {
            // Partial session event, exited child, SELinux denial, or unsupported procfs.
            return null
        }
    }

    private fun field(lines: List<String>, name: String) =
        lines.firstOrNull { it.startsWith("$name:") }?.substringAfter(':')?.trim()

    private companion object { val whitespace = Regex("\\s+") }
}
