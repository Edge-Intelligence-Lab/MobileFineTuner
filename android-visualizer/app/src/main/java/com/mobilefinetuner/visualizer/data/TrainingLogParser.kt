package com.mobilefinetuner.visualizer.data

import com.mobilefinetuner.visualizer.model.EventType
import com.mobilefinetuner.visualizer.model.RssPoint
import com.mobilefinetuner.visualizer.model.RunEvent
import com.mobilefinetuner.visualizer.model.RunStatus
import com.mobilefinetuner.visualizer.model.RunSummary
import com.mobilefinetuner.visualizer.model.StepMetric

object TrainingLogParser {

    data class ParsedTrainLog(
        val metrics: List<StepMetric>,
        val events: List<RunEvent>,
        val summary: RunSummary,
        val status: RunStatus
    )

    private data class MutableMetric(
        val step: Int,
        var totalSteps: Int? = null,
        var loss: Double? = null,
        var ppl: Double? = null,
        var lr: Double? = null,
        var tokens: Int? = null,
        var source: String = "log"
    ) {
        fun toImmutable(): StepMetric {
            return StepMetric(
                step = step,
                totalSteps = totalSteps,
                loss = loss,
                ppl = ppl,
                lr = lr,
                tokens = tokens,
                source = source
            )
        }
    }

    private val gpt2TrainPattern = Regex(
        """\[Train\].*?global\s+(\d+)/(\d+).*?\|\s*lr\s*([-+0-9.eE]+)\s*\|\s*loss\s*([-+0-9.eE]+)\s*\|\s*ppl\s*([-+0-9.eE]+).*?tokens\s+(\d+)"""
    )

    private val gemmaTrainPattern = Regex(
        """\[Step\s+(\d+)\]\s+Loss=([-+0-9.eE]+)\s+PPL=([-+0-9.eE]+)\s+LR=([-+0-9.eE]+)"""
    )

    private val rssSummaryPattern = Regex(
        """\[RSSSummary\]\s+MaxRSS\((init|train|step_end)\)\s*=\s*([-+0-9.eE]+)\s*MB"""
    )

    private val finalEmaPattern = Regex(
        """(?:final\s*ema|ema)\s*loss[:]?\s*([-+0-9.eE]+)""",
        RegexOption.IGNORE_CASE
    )
    private val totalTokensPattern = Regex(
        """(?:total\s*tokens|tokens|\u603btokens)[:]?\s*(\d+)""",
        RegexOption.IGNORE_CASE
    )
    private val totalStepsPattern = Regex(
        """(?:total\s*steps|steps|\u603b\u6b65\u6570)[:]?\s*(\d+)""",
        RegexOption.IGNORE_CASE
    )
    private val stepInEventPattern = Regex("""(?:global\s+|step=)(\d+)""")
    private const val completeCnMarker = "\u8bad\u7ec3\u5b8c\u6210"

    fun parseTrainLog(lines: List<String>): ParsedTrainLog {
        val metrics = linkedMapOf<Int, MutableMetric>()
        val events = mutableListOf<RunEvent>()

        var maxInitRss: Double? = null
        var maxTrainRss: Double? = null
        var maxStepEndRss: Double? = null
        var finalEmaLoss: Double? = null
        var totalTokens: Long? = null
        var totalSteps: Int? = null

        var hasCompletionMarker = false
        var hasFailureMarker = false

        for (line in lines) {
            parseTrainLine(line, metrics)

            val rssMatch = rssSummaryPattern.find(line)
            if (rssMatch != null) {
                val group = rssMatch.groupValues[1]
                val value = rssMatch.groupValues[2].toDoubleOrNull()
                when (group) {
                    "init" -> maxInitRss = value
                    "train" -> maxTrainRss = value
                    "step_end" -> maxStepEndRss = value
                }
            }

            finalEmaPattern.find(line)?.groupValues?.getOrNull(1)?.toDoubleOrNull()?.let {
                finalEmaLoss = it
            }
            totalTokensPattern.find(line)?.groupValues?.getOrNull(1)?.toLongOrNull()?.let {
                totalTokens = it
            }
            totalStepsPattern.find(line)?.groupValues?.getOrNull(1)?.toIntOrNull()?.let {
                totalSteps = it
            }

            val lower = line.lowercase()
            if (
                "training completed" in lower ||
                "lora training completed" in lower ||
                completeCnMarker in line
            ) {
                hasCompletionMarker = true
            }
            if ("error" in lower || "failed" in lower) {
                hasFailureMarker = true
            }

            classifyEvent(line)?.let { type ->
                val step = stepInEventPattern.find(line)?.groupValues?.getOrNull(1)?.toIntOrNull()
                events += RunEvent(
                    step = step,
                    type = type,
                    message = line,
                    raw = line
                )
            }
        }

        val mergedMetrics = metrics.values
            .map { it.toImmutable() }
            .sortedBy { it.step }

        val status = when {
            hasFailureMarker -> RunStatus.FAILED
            hasCompletionMarker -> RunStatus.COMPLETED
            mergedMetrics.isNotEmpty() -> RunStatus.RUNNING
            else -> RunStatus.IDLE
        }

        val summary = RunSummary(
            maxInitRssMb = maxInitRss,
            maxTrainRssMb = maxTrainRss,
            maxStepEndRssMb = maxStepEndRss,
            finalEmaLoss = finalEmaLoss,
            totalTokens = totalTokens,
            totalSteps = totalSteps
        )

        return ParsedTrainLog(
            metrics = mergedMetrics,
            events = events,
            summary = summary,
            status = status
        )
    }

    private fun parseTrainLine(line: String, metrics: MutableMap<Int, MutableMetric>) {
        gpt2TrainPattern.find(line)?.let { match ->
            val step = match.groupValues[1].toIntOrNull() ?: return@let
            val total = match.groupValues[2].toIntOrNull()
            val lr = match.groupValues[3].toDoubleOrNull()
            val loss = match.groupValues[4].toDoubleOrNull()
            val ppl = match.groupValues[5].toDoubleOrNull()
            val tokens = match.groupValues[6].toIntOrNull()
            val metric = metrics.getOrPut(step) { MutableMetric(step) }
            metric.totalSteps = total ?: metric.totalSteps
            metric.lr = lr ?: metric.lr
            metric.loss = loss ?: metric.loss
            metric.ppl = ppl ?: metric.ppl
            metric.tokens = tokens ?: metric.tokens
            metric.source = "gpt2"
            return
        }

        gemmaTrainPattern.find(line)?.let { match ->
            val step = match.groupValues[1].toIntOrNull() ?: return@let
            val loss = match.groupValues[2].toDoubleOrNull()
            val ppl = match.groupValues[3].toDoubleOrNull()
            val lr = match.groupValues[4].toDoubleOrNull()
            val metric = metrics.getOrPut(step) { MutableMetric(step) }
            metric.loss = loss ?: metric.loss
            metric.ppl = ppl ?: metric.ppl
            metric.lr = lr ?: metric.lr
            metric.source = "gemma"
        }
    }

    private fun classifyEvent(line: String): EventType? {
        val lower = line.lowercase()
        return when {
            "[eval]" in lower -> EventType.EVAL
            "[checkpoint]" in lower -> EventType.CHECKPOINT
            "[memcleanup]" in lower -> EventType.CLEANUP
            "[rsssummary]" in lower || "[rssstep]" in lower -> EventType.RSS
            "[energy]" in lower || "sleep_ms" in lower -> EventType.ENERGY
            "⚠" in line || "warn" in lower -> EventType.WARNING
            "error" in lower || "failed" in lower -> EventType.ERROR
            "[train]" in lower || "[step " in lower -> EventType.INFO
            else -> null
        }
    }

    fun parseRssCsv(lines: List<String>): List<RssPoint> {
        if (lines.isEmpty()) return emptyList()
        val header = lines.first().split(',').map { it.trim() }
        val hasHeader = header.any { it.equals("rss_mb", ignoreCase = true) }

        val rssIndex = if (hasHeader) {
            header.indexOfFirst { it.equals("rss_mb", ignoreCase = true) }.coerceAtLeast(0)
        } else {
            2
        }
        val tickIndex = if (hasHeader) {
            header.indexOfFirst { it.equals("tick", ignoreCase = true) || it.equals("step", ignoreCase = true) }
        } else {
            0
        }
        val tsIndex = if (hasHeader) {
            header.indexOfFirst { it.equals("timestamp", ignoreCase = true) }
        } else {
            -1
        }

        val start = if (hasHeader) 1 else 0
        val points = mutableListOf<RssPoint>()

        for (row in lines.drop(start)) {
            if (row.isBlank()) continue
            val cols = row.split(',')
            val index = if (tickIndex in cols.indices) {
                cols[tickIndex].trim().toIntOrNull() ?: points.size
            } else {
                points.size
            }
            val rss = if (rssIndex in cols.indices) {
                cols[rssIndex].trim().toDoubleOrNull()
            } else {
                null
            }
            if (rss != null) {
                val ts = if (tsIndex in cols.indices) cols[tsIndex].trim() else null
                points += RssPoint(index = index, rssMb = rss, timeLabel = ts)
            }
        }

        return points
    }

    fun parseMetricsNdjson(lines: List<String>): List<StepMetric> {
        val metrics = mutableListOf<StepMetric>()

        for (line in lines) {
            val trimmed = line.trim()
            if (trimmed.isEmpty() || !trimmed.startsWith("{")) continue

            val step = findInt(trimmed, "step") ?: continue
            val metric = StepMetric(
                step = step,
                totalSteps = findInt(trimmed, "total_steps"),
                loss = findDouble(trimmed, "loss"),
                ppl = findDouble(trimmed, "ppl"),
                lr = findDouble(trimmed, "lr"),
                tokens = findInt(trimmed, "tokens"),
                source = "ndjson"
            )
            metrics += metric
        }

        return metrics.sortedBy { it.step }
    }

    private fun findDouble(line: String, key: String): Double? {
        val pattern = Regex("\"$key\"\\s*:\\s*([-+0-9.eE]+)")
        return pattern.find(line)?.groupValues?.getOrNull(1)?.toDoubleOrNull()
    }

    private fun findInt(line: String, key: String): Int? {
        val pattern = Regex("\"$key\"\\s*:\\s*([-+0-9]+)")
        return pattern.find(line)?.groupValues?.getOrNull(1)?.toIntOrNull()
    }
}
