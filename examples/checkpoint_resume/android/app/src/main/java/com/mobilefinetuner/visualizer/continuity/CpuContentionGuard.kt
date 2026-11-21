package com.mobilefinetuner.visualizer.continuity

internal data class SchedulerTimes(val runNs: Long, val waitNs: Long)

/** Leader-only run-queue delay. Sleeping (including I/O) is not runnable waiting. */
internal class CpuContentionGuard {
    private var base: SchedulerTimes? = null
    private var baseMs = 0L
    private var sustainedMs = 0L
    private var nextReadMs = 0L

    fun poll(nowMs: Long, eligible: Boolean, read: () -> SchedulerTimes?): String? {
        if (!eligible) {
            nextReadMs = 0
            return observe(nowMs, null)
        }
        if (nowMs < nextReadMs) return null
        nextReadMs = nowMs + 1000
        return observe(nowMs, read())
    }

    fun observe(nowMs: Long, times: SchedulerTimes?): String? {
        val old = base
        val span = nowMs - baseMs
        if (times == null || old == null || span !in 0..2000 ||
            times.runNs < old.runNs || times.waitNs < old.waitNs) {
            base = times
            baseMs = nowMs
            sustainedMs = 0
            return null
        }
        if (span < 1000) return null
        base = times
        baseMs = nowMs
        val run = (times.runNs - old.runNs) / 1e6
        val wait = (times.waitNs - old.waitNs) / 1e6
        val runnable = wait + run
        // Ignore mostly sleeping windows and implausible deltas (e.g. stale accounting).
        val share = if (runnable > 0) wait / runnable else 0.0
        val threshold = if (sustainedMs == 0L) 0.45 else 0.35
        if (runnable < span * 0.25 || runnable > span * 1.20 || wait < span * 0.15 || share < threshold) {
            sustainedMs = 0
            return null
        }
        sustainedMs += span
        return if (sustainedMs >= 3000)
            "CPU contention: trainer leader runnable wait ${wait.toInt()} ms, running ${run.toInt()} ms " +
                "in ${span} ms; sustained ${sustainedMs} ms."
        else null
    }
}
