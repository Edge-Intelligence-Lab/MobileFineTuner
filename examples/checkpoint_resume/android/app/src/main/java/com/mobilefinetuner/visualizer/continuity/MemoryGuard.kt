package com.mobilefinetuner.visualizer.continuity

/** All sizes are MiB. The entry margin avoids restarting right at the yield threshold. */
internal class MemoryGuard(private val minimum: Int) {
    private var belowSince: Long? = null

    private fun floor(r: Resources) = maxOf(minimum.toLong(), r.lowThresholdMb + 128,
        minOf(r.totalMb / 8, 1024L))
    private fun reason(r: Resources, limit: Long) =
        "Memory pressure: available ${r.availableMb} MiB; reserve ${limit} MiB; Android lowMemory=${r.lowMemory}."

    fun beforeStart(r: Resources): String? {
        val limit = floor(r) + 128
        return if (r.lowMemory || r.availableMb < limit) reason(r, limit) else null
    }

    fun duringTraining(r: Resources, elapsedMs: Long): String? {
        val limit = floor(r)
        if (r.lowMemory || r.availableMb < maxOf(minimum.toLong(), r.lowThresholdMb))
            return reason(r, limit)
        if (r.availableMb >= limit) {
            belowSince = null
            return null
        }
        val start = belowSince ?: elapsedMs.also { belowSince = it }
        return if (elapsedMs - start >= 1000) reason(r, limit) else null
    }
}
