package com.mobilefinetuner.visualizer.ui.components

import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.CornerRadius
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.lerp
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.drawscope.drawIntoCanvas
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.unit.dp
import com.mobilefinetuner.visualizer.model.RssPoint
import com.mobilefinetuner.visualizer.model.StepMetric
import com.mobilefinetuner.visualizer.ui.theme.NeonAmber
import com.mobilefinetuner.visualizer.ui.theme.NeonBlue
import java.util.Locale
import kotlin.math.ceil
import kotlin.math.max

private data class DynamicPhase(
    val index: Int,
    val label: String,
    val startStep: Int,
    val endStep: Int,
    val avgRssMb: Double?,
    val minRssMb: Double?,
    val maxRssMb: Double?,
    val avgLoss: Double?,
    val avgPpl: Double?,
    val avgLr: Double?,
    val lossDeltaVariance: Double?,
    val barColor: Color
)

private val phasePalette = listOf(
    Color(0xFFA0A7B4), // graphite
    Color(0xFF4A8FE8), // blue
    Color(0xFF5EA5ED), // sky
    Color(0xFF5CC0B5), // aqua
    Color(0xFF6DBE8C), // mint
    Color(0xFF7D8CE7), // indigo
    Color(0xFF5F89D4), // steel blue
    Color(0xFF6DA6CF), // cool teal
    Color(0xFF76B4A2), // seafoam
    Color(0xFF8D9EBD)  // slate
)

private val lossPhaseColor = Color(0xFFE56366)
private val pplPhaseColor = Color(0xFF3C8DFF)
private val lrPhaseColor = Color(0xFF30A87A)
private val varPhaseColor = Color(0xFFE39A3B)

@Composable
fun MemoryWaterfallChart(
    modifier: Modifier = Modifier,
    metrics: List<StepMetric>,
    rssPoints: List<RssPoint>
) {
    val phases = remember(metrics, rssPoints) {
        buildDynamicPhases(metrics = metrics, rssPoints = rssPoints)
    }

    if (phases.isEmpty()) {
        Column(
            modifier = modifier
                .background(
                    brush = Brush.verticalGradient(listOf(Color(0xFFFFFFFF), Color(0xFFF8F8FF))),
                    shape = RoundedCornerShape(22.dp)
                )
                .padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            Text(
                text = "Realtime Memory Dynamics",
                style = MaterialTheme.typography.titleLarge,
                color = MaterialTheme.colorScheme.onSurface
            )
            Text(
                text = "Not enough training points to build dynamic windows yet.",
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
            Text(
                text = "Tip: keep training running and provide rss.csv for richer memory-phase insight.",
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.primary.copy(alpha = 0.9f)
            )
        }
        return
    }

    val rssValues = phases.mapNotNull { it.avgRssMb }
    val maxVal = (rssValues.maxOrNull() ?: 1.0).toFloat()
    val minVal = (rssValues.minOrNull() ?: 0.0).toFloat()
    val span = max(1f, maxVal - minVal)
    var selectedBarIndex by remember(phases) { mutableIntStateOf(phases.lastIndex) }

    val firstRss = phases.firstOrNull { it.avgRssMb != null }?.avgRssMb
    val lastRss = phases.lastOrNull { it.avgRssMb != null }?.avgRssMb
    val summaryLine = if (firstRss != null && lastRss != null) {
        val deltaPct = ((lastRss - firstRss) / firstRss) * 100.0
        "Windowed RSS: ${format1(firstRss)} MB → ${format1(lastRss)} MB  •  ${formatSigned1(deltaPct)}%  •  Tap a phase"
    } else {
        "RSS stream missing (rss.csv not found). Tap a phase to inspect loss/LR dynamics."
    }

    Column(
        modifier = modifier
            .background(
                brush = Brush.verticalGradient(
                    listOf(Color(0xFFFFFFFF), Color(0xFFF8F8FF))
                ),
                shape = RoundedCornerShape(22.dp)
            )
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp)
    ) {
        // ── Header ────────────────────────────────────────────────────────
        Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
            Text(
                text  = "Realtime Memory Dynamics",
                style = MaterialTheme.typography.titleLarge,
                color = MaterialTheme.colorScheme.onSurface
            )
            Text(
                text  = summaryLine,
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
        }

        // ── Bar chart ─────────────────────────────────────────────────────
        Canvas(
            modifier = Modifier
                .fillMaxWidth()
                .height(210.dp)
                .pointerInput(phases) {
                    detectTapGestures { offset ->
                        val totalW    = size.width.toFloat()
                        val barSpacing = 10f
                        val barW      = (totalW - barSpacing * (phases.size + 1)) / phases.size
                        val tappedIdx = ((offset.x - barSpacing) / (barW + barSpacing))
                            .toInt()
                            .coerceIn(0, phases.lastIndex)
                        selectedBarIndex = if (selectedBarIndex == tappedIdx) -1 else tappedIdx
                    }
                }
        ) {
            val barCount   = phases.size
            val totalW     = size.width
            val totalH     = size.height
            val barSpacing = 10f
            val labelH     = 44f
            val valueH     = 30f
            val chartH     = totalH - labelH - valueH
            val barW       = (totalW - barSpacing * (barCount + 1)) / barCount

            phases.forEachIndexed { i, stage ->
                val rss = stage.avgRssMb?.toFloat()
                val frac = if (rss != null) ((rss - minVal) / span).coerceIn(0f, 1f) else 0f
                val barH       = chartH * frac
                val x          = barSpacing + i * (barW + barSpacing)
                val y          = valueH + (chartH - barH)
                val isSelected = i == selectedBarIndex

                // Main bar (fully opaque when selected)
                drawRoundRect(
                    color        = if (isSelected) stage.barColor else stage.barColor.copy(alpha = 0.80f),
                    topLeft      = Offset(x, y),
                    size         = Size(barW, barH.coerceAtLeast(6f)),
                    cornerRadius = CornerRadius(8f, 8f)
                )

                // Selection ring
                if (isSelected) {
                    drawRoundRect(
                        color        = stage.barColor,
                        topLeft      = Offset(x - 2f, y - 2f),
                        size         = Size(barW + 4f, barH + 2f),
                        cornerRadius = CornerRadius(10f, 10f),
                        style        = Stroke(width = 3f)
                    )
                }

                // Shimmer on top half
                drawRoundRect(
                    brush        = Brush.verticalGradient(
                        colors = listOf(Color.White.copy(alpha = 0.35f), Color.Transparent),
                        startY = y,
                        endY   = y + barH.coerceAtLeast(6f) / 2f
                    ),
                    topLeft      = Offset(x, y),
                    size         = Size(barW, barH.coerceAtLeast(6f) / 2f),
                    cornerRadius = CornerRadius(8f, 8f)
                )

                // Connector line to next bar
                if (i < phases.size - 1) {
                    val nextVal = phases[i + 1].avgRssMb?.toFloat()
                    val nextFrac = if (nextVal != null) ((nextVal - minVal) / span).coerceIn(0f, 1f) else 0f
                    val nextH    = chartH * nextFrac
                    val nextX    = barSpacing + (i + 1) * (barW + barSpacing)
                    val startY   = valueH + (chartH - barH) + barH.coerceAtLeast(6f) / 2f
                    val endY     = valueH + (chartH - nextH) + nextH.coerceAtLeast(6f) / 2f
                    drawLine(
                        color       = Color(0xFFC6C6C8),
                        start       = Offset(x + barW, startY),
                        end         = Offset(nextX, endY),
                        strokeWidth = 1.5f,
                        cap         = StrokeCap.Round
                    )
                }

                // MB value label above bar
                drawIntoCanvas { canvas ->
                    val paint = android.graphics.Paint().apply {
                        color          = if (isSelected)
                            android.graphics.Color.argb(255, 0, 0, 0)
                        else
                            android.graphics.Color.argb(200, 0, 0, 0)
                        textSize       = 24f
                        textAlign      = android.graphics.Paint.Align.CENTER
                        isFakeBoldText = isSelected
                    }
                    canvas.nativeCanvas.drawText(
                        stage.avgRssMb?.let { formatInt(it) } ?: "N/A",
                        x + barW / 2f,
                        y - 8f,
                        paint
                    )
                }

                // X-axis label below bar (blue + bold when selected)
                drawIntoCanvas { canvas ->
                    val paint = android.graphics.Paint().apply {
                        color = if (isSelected)
                            android.graphics.Color.argb(255, 0, 80, 200)
                        else
                            android.graphics.Color.argb(180, 100, 100, 112)
                        textSize       = 22f
                        textAlign      = android.graphics.Paint.Align.CENTER
                        isFakeBoldText = isSelected
                    }
                    canvas.nativeCanvas.drawText(
                        stage.label,
                        x + barW / 2f,
                        totalH - 6f,
                        paint
                    )
                }
            }

            // ── Dynamic low-pressure dashed line ───────────────────────────
            if (rssValues.isNotEmpty()) {
                val lowPressure = (minVal + span * 0.2f).coerceAtMost(maxVal)
                val thresholdY = valueH + chartH * (1f - ((lowPressure - minVal) / span))
                val dashLen = 14f
                var dashX = 0f
                while (dashX < totalW) {
                    drawLine(
                        color       = Color(0xFFC9A266).copy(alpha = 0.72f),
                        start       = Offset(dashX, thresholdY),
                        end         = Offset(minOf(dashX + dashLen, totalW), thresholdY),
                        strokeWidth = 1.5f
                    )
                    dashX += dashLen * 2f
                }
            }
        }

        // ── Selected stage detail panel ───────────────────────────────────
        AnimatedVisibility(
            visible = selectedBarIndex in phases.indices,
            enter   = fadeIn(),
            exit    = fadeOut()
        ) {
            val sel = phases.getOrNull(selectedBarIndex)
            if (sel != null) {
                val base = phases.firstOrNull { it.avgRssMb != null }?.avgRssMb
                val selRss = sel.avgRssMb
                val reductionPct = if (base != null && selRss != null && base > 0.0) {
                    (1.0 - (selRss / base)) * 100.0
                } else {
                    null
                }
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .background(sel.barColor.copy(alpha = 0.12f), RoundedCornerShape(12.dp))
                        .padding(horizontal = 14.dp, vertical = 10.dp),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment     = Alignment.CenterVertically
                ) {
                    Column(verticalArrangement = Arrangement.spacedBy(2.dp)) {
                        Text(
                            text  = "Phase ${sel.label}  •  step ${sel.startStep}–${sel.endStep}",
                            style = MaterialTheme.typography.labelLarge,
                            color = sel.barColor
                        )
                        if (reductionPct != null) {
                            Text(
                                text  = "${formatSigned1(reductionPct)}% vs first phase",
                                style = MaterialTheme.typography.bodyMedium,
                                color = sel.barColor.copy(alpha = 0.75f)
                            )
                        }
                    }
                    Text(
                        text  = sel.avgRssMb?.let { "${format1(it)} MB" } ?: "RSS N/A",
                        style = MaterialTheme.typography.titleMedium,
                        color = sel.barColor
                    )
                }
            }
        }

        // ── Per-phase metric chips ───────────────────────────────────────
        phases.getOrNull(selectedBarIndex)?.let { sel ->
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                PhaseChip(
                    title = "Loss",
                    value = sel.avgLoss?.let { format3(it) } ?: "N/A",
                    color = lossPhaseColor,
                    modifier = Modifier.weight(1f)
                )
                PhaseChip(
                    title = "PPL",
                    value = sel.avgPpl?.let { format2(it) } ?: "N/A",
                    color = pplPhaseColor,
                    modifier = Modifier.weight(1f)
                )
                PhaseChip(
                    title = "LR",
                    value = sel.avgLr?.let { format6(it) } ?: "N/A",
                    color = lrPhaseColor,
                    modifier = Modifier.weight(1f)
                )
                PhaseChip(
                    title = "Var",
                    value = sel.lossDeltaVariance?.let { format4(it) } ?: "N/A",
                    color = varPhaseColor,
                    modifier = Modifier.weight(1f)
                )
            }
        }

        // ── LR + volatility heatstrip ────────────────────────────────────
        Text(
            text = "LR / Volatility Heatstrip (tap to focus phase)",
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )
        Canvas(
            modifier = Modifier
                .fillMaxWidth()
                .height(72.dp)
                .pointerInput(phases) {
                    detectTapGestures { offset ->
                        val cellW = size.width / phases.size
                        val idx = (offset.x / cellW).toInt().coerceIn(0, phases.lastIndex)
                        selectedBarIndex = idx
                    }
                }
        ) {
            val lrVals = phases.mapNotNull { it.avgLr }
            val varVals = phases.mapNotNull { it.lossDeltaVariance }
            val minLr = lrVals.minOrNull() ?: 0.0
            val maxLr = lrVals.maxOrNull() ?: 1.0
            val minVar = varVals.minOrNull() ?: 0.0
            val maxVar = varVals.maxOrNull() ?: 1.0
            val w = size.width / phases.size
            val h = size.height
            phases.forEachIndexed { i, phase ->
                val x = i * w
                val lrNorm = normalize(phase.avgLr, minLr, maxLr)
                val varNorm = normalize(phase.lossDeltaVariance, minVar, maxVar)
                val top = lerp(Color(0xFFE7EEF9), Color(0xFF4A8FE8), lrNorm.toFloat())
                val bottom = lerp(Color(0xFFF4EEE5), Color(0xFFE39A3B), varNorm.toFloat())
                drawRoundRect(
                    color = top,
                    topLeft = Offset(x + 2f, 2f),
                    size = Size(w - 4f, h / 2f - 3f),
                    cornerRadius = CornerRadius(6f, 6f)
                )
                drawRoundRect(
                    color = bottom,
                    topLeft = Offset(x + 2f, h / 2f + 1f),
                    size = Size(w - 4f, h / 2f - 3f),
                    cornerRadius = CornerRadius(6f, 6f)
                )
                if (i == selectedBarIndex) {
                    drawRoundRect(
                        color = phase.barColor,
                        topLeft = Offset(x + 1f, 1f),
                        size = Size(w - 2f, h - 2f),
                        cornerRadius = CornerRadius(8f, 8f),
                        style = Stroke(width = 2f)
                    )
                }
            }
        }

        // ── Legend row ────────────────────────────────────────────────────
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(10.dp)
        ) {
            LegendCell(color = Color(0xFF4A8FE8), label = "Higher LR")
            LegendCell(color = Color(0xFFE39A3B), label = "Higher Volatility")
            LegendCell(color = phases.getOrNull(selectedBarIndex)?.barColor ?: Color(0xFF6D8FD4), label = "Selected Phase")
        }
    }
}

@Composable
private fun LegendCell(color: Color, label: String) {
    Row(
        horizontalArrangement = Arrangement.spacedBy(6.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Box(
            modifier = Modifier
                .width(16.dp)
                .height(4.dp)
                .background(color, RoundedCornerShape(2.dp))
        )
        Text(
            text = label,
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )
    }
}

@Composable
private fun PhaseChip(
    title: String,
    value: String,
    color: Color,
    modifier: Modifier = Modifier
) {
    Column(
        modifier = modifier
            .background(color.copy(alpha = 0.10f), RoundedCornerShape(10.dp))
            .padding(horizontal = 8.dp, vertical = 6.dp)
    ) {
        Text(
            text = title,
            style = MaterialTheme.typography.labelSmall,
            color = color.copy(alpha = 0.85f)
        )
        Text(
            text = value,
            style = MaterialTheme.typography.bodyLarge,
            color = color
        )
    }
}

private fun buildDynamicPhases(
    metrics: List<StepMetric>,
    rssPoints: List<RssPoint>
): List<DynamicPhase> {
    val n = max(metrics.size, rssPoints.size)
    if (n < 3) return emptyList()

    val targetWindows = when {
        n >= 180 -> 8
        n >= 120 -> 7
        n >= 60 -> 6
        n >= 30 -> 5
        else -> 4
    }

    val windowSize = ceil(n / targetWindows.toDouble()).toInt().coerceAtLeast(1)
    val phases = mutableListOf<DynamicPhase>()
    var idx = 0
    var phaseIdx = 0
    while (idx < n) {
        val endExclusive = (idx + windowSize).coerceAtMost(n)
        val metricSlice = metrics.subList(idx.coerceAtMost(metrics.size), endExclusive.coerceAtMost(metrics.size))
        val rssSlice = rssPoints.subList(idx.coerceAtMost(rssPoints.size), endExclusive.coerceAtMost(rssPoints.size))
            .map { it.rssMb }

        val losses = metricSlice.mapNotNull { it.loss }
        val deltas = losses.zipWithNext { a, b -> b - a }
        val variance = deltas.takeIf { it.size >= 2 }?.let { sampleVariance(it) }

        val startStep = metrics.getOrNull(idx)?.step ?: (idx + 1)
        val endStep = metrics.getOrNull((endExclusive - 1).coerceAtLeast(0))?.step ?: endExclusive

        phases += DynamicPhase(
            index = phaseIdx,
            label = "P${phaseIdx + 1}",
            startStep = startStep,
            endStep = endStep,
            avgRssMb = rssSlice.takeIf { it.isNotEmpty() }?.average(),
            minRssMb = rssSlice.minOrNull(),
            maxRssMb = rssSlice.maxOrNull(),
            avgLoss = losses.takeIf { it.isNotEmpty() }?.average(),
            avgPpl = metricSlice.mapNotNull { it.ppl }.takeIf { it.isNotEmpty() }?.average(),
            avgLr = metricSlice.mapNotNull { it.lr }.takeIf { it.isNotEmpty() }?.average(),
            lossDeltaVariance = variance,
            barColor = phasePalette[phaseIdx % phasePalette.size]
        )

        idx = endExclusive
        phaseIdx += 1
    }

    return phases
}

private fun sampleVariance(values: List<Double>): Double {
    if (values.size < 2) return 0.0
    val mean = values.average()
    val sum = values.sumOf { v ->
        val d = v - mean
        d * d
    }
    return sum / (values.size - 1).toDouble()
}

private fun normalize(value: Double?, min: Double, max: Double): Double {
    if (value == null) return 0.0
    val span = (max - min).takeIf { it > 1e-12 } ?: return 0.0
    return ((value - min) / span).coerceIn(0.0, 1.0)
}

private fun formatInt(v: Double): String = String.format(Locale.US, "%.0f", v)
private fun format1(v: Double): String = String.format(Locale.US, "%.1f", v)
private fun format2(v: Double): String = String.format(Locale.US, "%.2f", v)
private fun format3(v: Double): String = String.format(Locale.US, "%.3f", v)
private fun format4(v: Double): String = String.format(Locale.US, "%.4f", v)
private fun format6(v: Double): String = String.format(Locale.US, "%.6f", v)
private fun formatSigned1(v: Double): String = if (v >= 0.0) "+${format1(v)}" else format1(v)
