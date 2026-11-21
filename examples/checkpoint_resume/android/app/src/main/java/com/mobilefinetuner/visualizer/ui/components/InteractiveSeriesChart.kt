package com.mobilefinetuner.visualizer.ui.components

import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.tween
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.gestures.detectHorizontalDragGestures
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.drawscope.drawIntoCanvas
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.unit.dp
import java.util.Locale
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

data class LineSeries(
    val name: String,
    val values: List<Double>,
    val color: Color
)

@Composable
fun InteractiveMetricLineChart(
    title: String,
    subtitle: String,
    values: List<Double>,
    color: Color,
    modifier: Modifier = Modifier,
    seriesLabel: String = title,
    valueFormatter: (Double) -> String = { String.format(Locale.US, "%.3f", it) }
) {
    InteractiveSeriesChart(
        title = title,
        subtitle = subtitle,
        series = listOf(LineSeries(seriesLabel, values, color)),
        modifier = modifier,
        valueFormatter = valueFormatter
    )
}

@Composable
fun InteractiveComparisonChart(
    title: String,
    subtitle: String,
    baselineName: String,
    baselineValues: List<Double>,
    candidateName: String,
    candidateValues: List<Double>,
    baselineColor: Color,
    candidateColor: Color,
    modifier: Modifier = Modifier,
    valueFormatter: (Double) -> String = { String.format(Locale.US, "%.3f", it) }
) {
    InteractiveSeriesChart(
        title = title,
        subtitle = subtitle,
        series = listOf(
            LineSeries(baselineName, baselineValues, baselineColor),
            LineSeries(candidateName, candidateValues, candidateColor)
        ),
        modifier = modifier,
        valueFormatter = valueFormatter
    )
}

// ── Catmull-Rom bezier control points ─────────────────────────────────────
private fun catmullRomCP(
    p0: Offset, p1: Offset, p2: Offset, p3: Offset
): Pair<Offset, Offset> {
    val t = 0.4f
    val c1 = Offset(
        p1.x + (p2.x - p0.x) * t / 2f,
        p1.y + (p2.y - p0.y) * t / 2f
    )
    val c2 = Offset(
        p2.x - (p3.x - p1.x) * t / 2f,
        p2.y - (p3.y - p1.y) * t / 2f
    )
    return c1 to c2
}

@Composable
private fun InteractiveSeriesChart(
    title: String,
    subtitle: String,
    series: List<LineSeries>,
    modifier: Modifier = Modifier,
    valueFormatter: (Double) -> String
) {
    val chartSeries = remember(series) {
        series.map { s ->
            s.copy(values = downsampleEvenly(s.values, maxPoints = 360))
        }
    }
    val maxLen = chartSeries.maxOfOrNull { it.values.size } ?: 0
    val onSurfaceColor = MaterialTheme.colorScheme.onSurface

    var zoomX by remember(title, maxLen) { mutableFloatStateOf(1f) }
    var panNorm by remember(title, maxLen) { mutableFloatStateOf(0f) }
    var selectedGlobalIndex by remember(title, maxLen) { mutableIntStateOf(-1) }

    val reveal by animateFloatAsState(
        targetValue = 1f,
        animationSpec = tween(durationMillis = 750),
        label = "chartReveal"
    )

    val visibleCount = if (maxLen <= 0) 0 else max(8, (maxLen / zoomX).roundToInt())
    val maxStart = max(0, maxLen - visibleCount)
    val startIndex = (panNorm * maxStart).roundToInt().coerceIn(0, maxStart)
    val endIndex = min(maxLen, startIndex + visibleCount)
    val yAxisPaint = remember {
        android.graphics.Paint().apply {
            textSize = 26f
            textAlign = android.graphics.Paint.Align.RIGHT
            typeface = android.graphics.Typeface.MONOSPACE
        }
    }
    val xAxisPaint = remember {
        android.graphics.Paint().apply {
            textSize = 24f
            textAlign = android.graphics.Paint.Align.CENTER
            typeface = android.graphics.Typeface.MONOSPACE
        }
    }

    Column(
        modifier = modifier
            .background(
                brush = Brush.verticalGradient(
                    listOf(
                        MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.92f),
                        MaterialTheme.colorScheme.surface.copy(alpha = 0.98f)
                    )
                ),
                shape = RoundedCornerShape(22.dp)
            )
            .padding(14.dp),
        verticalArrangement = Arrangement.spacedBy(10.dp)
    ) {
        // ── Header row: title + legend ────────────────────────────────────
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.Top
        ) {
            Column(verticalArrangement = Arrangement.spacedBy(2.dp)) {
                Text(
                    text = title,
                    style = MaterialTheme.typography.titleLarge,
                    color = MaterialTheme.colorScheme.onSurface
                )
                Text(
                    text = subtitle,
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }
            // Series legend with colored dot + name + last value
            Column(
                horizontalAlignment = Alignment.End,
                verticalArrangement = Arrangement.spacedBy(4.dp)
            ) {
                chartSeries.forEach { s ->
                    val value = s.values.lastOrNull()?.let(valueFormatter) ?: "N/A"
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(4.dp)
                    ) {
                        Box(
                            modifier = Modifier
                                .size(8.dp)
                                .background(s.color, CircleShape)
                        )
                        Text(
                            text = "${s.name}: $value",
                            style = MaterialTheme.typography.bodyMedium,
                            color = s.color
                        )
                    }
                }
            }
        }

        // ── Chart canvas ──────────────────────────────────────────────────
        Canvas(
            modifier = Modifier
                .fillMaxWidth()
                .height(220.dp)
                .pointerInput(maxLen) {
                    detectHorizontalDragGestures { _, dragAmount ->
                        if (maxLen <= 1) return@detectHorizontalDragGestures
                        val nextVisible = max(8, (maxLen / zoomX).roundToInt())
                        val nextMaxStart = max(0, maxLen - nextVisible)
                        if (nextMaxStart == 0) return@detectHorizontalDragGestures

                        val delta = -dragAmount / size.width
                        panNorm = (panNorm + delta / zoomX).coerceIn(0f, 1f)
                    }
                }
                .pointerInput(maxLen) {
                    detectTapGestures(
                        onDoubleTap = {
                            zoomX = when {
                                zoomX < 1.5f -> 2f
                                zoomX < 3.5f -> 4f
                                else -> 1f
                            }
                            if (zoomX == 1f) panNorm = 0f
                            selectedGlobalIndex = -1
                        },
                        onTap = { offset ->
                            if (maxLen <= 1) return@detectTapGestures
                            val dynamicVisible = max(8, (maxLen / zoomX).roundToInt())
                            val dynamicMaxStart = max(0, maxLen - dynamicVisible)
                            val dynamicStart = (panNorm * dynamicMaxStart).roundToInt().coerceIn(0, dynamicMaxStart)
                            val dynamicEnd = min(maxLen, dynamicStart + dynamicVisible)
                            if (dynamicEnd <= dynamicStart) return@detectTapGestures
                            val leftPad = 52f
                            val rightPad = 12f
                            val chartWidth = (size.width - leftPad - rightPad).coerceAtLeast(1f)
                            val local = ((offset.x - leftPad) / chartWidth).coerceIn(0f, 1f)
                            val idxInWindow = (local * (dynamicEnd - dynamicStart - 1).coerceAtLeast(0)).roundToInt()
                            selectedGlobalIndex = (dynamicStart + idxInWindow).coerceIn(0, maxLen - 1)
                        }
                    )
                }
        ) {
            if (maxLen <= 1 || endIndex <= startIndex) {
                drawRect(color = onSurfaceColor.copy(alpha = 0.04f), size = size)
                return@Canvas
            }

            // Expanded padding to accommodate axis labels
            val leftPad   = 52f   // Y-axis label zone
            val rightPad  = 12f
            val topPad    = 18f
            val bottomPad = 36f   // X-axis label zone
            val chartWidth  = size.width - leftPad - rightPad
            val chartHeight = size.height - topPad - bottomPad

            val visibleSeries = chartSeries.map { s ->
                val visibleValues = if (s.values.isEmpty()) {
                    emptyList()
                } else {
                    val safeEnd = min(endIndex, s.values.size)
                    if (safeEnd <= startIndex) emptyList() else s.values.subList(startIndex, safeEnd)
                }
                s.copy(values = visibleValues)
            }

            val allVisibleValues = visibleSeries.flatMap { it.values }
            val maxValue = allVisibleValues.maxOrNull() ?: 1.0
            val minValue = allVisibleValues.minOrNull() ?: 0.0
            val range = max(1e-9, maxValue - minValue)

            // ── Horizontal grid lines ─────────────────────────────────────
            for (i in 0..4) {
                val y = topPad + chartHeight * i / 4f
                drawLine(
                    color = Color.Gray.copy(alpha = 0.12f),
                    start = Offset(leftPad, y),
                    end   = Offset(size.width - rightPad, y),
                    strokeWidth = 1f
                )
            }

            // ── Y-axis labels (dark text for light background) ────────────
            drawIntoCanvas { canvas ->
                yAxisPaint.color = android.graphics.Color.argb(180, 100, 100, 112)
                for (i in 0..4) {
                    val fraction = 1f - i / 4f
                    val yVal     = minValue + fraction * range
                    val y        = topPad + chartHeight * i / 4f + 9f
                    canvas.nativeCanvas.drawText(
                        valueFormatter(yVal),
                        leftPad - 6f,
                        y,
                        yAxisPaint
                    )
                }
            }

            // ── X-axis step labels ────────────────────────────────────────
            drawIntoCanvas { canvas ->
                xAxisPaint.color = android.graphics.Color.argb(150, 100, 100, 112)
                val xCount = 5
                for (i in 0 until xCount) {
                    val fraction = i / (xCount - 1).toFloat()
                    val x        = leftPad + chartWidth * fraction
                    val stepNum  = startIndex + ((endIndex - startIndex - 1).toFloat() * fraction).roundToInt()
                    canvas.nativeCanvas.drawText(
                        "s$stepNum",
                        x,
                        size.height - 6f,
                        xAxisPaint
                    )
                }
            }

            // ── Draw series with Catmull-Rom bezier smoothing ─────────────
            visibleSeries.forEach { s ->
                if (s.values.size < 2) return@forEach

                val effectiveSize = max(2, (s.values.size * reveal).roundToInt())
                val valuesToDraw  = s.values.take(effectiveSize)

                val path = Path()
                val fill = Path()

                fun point(i: Int): Offset {
                    val x          = leftPad + chartWidth * i / (valuesToDraw.size - 1).toFloat()
                    val normalized = ((valuesToDraw[i] - minValue) / range).toFloat()
                    val y          = topPad + chartHeight * (1f - normalized)
                    return Offset(x, y)
                }

                val first = point(0)
                path.moveTo(first.x, first.y)
                fill.moveTo(first.x, topPad + chartHeight)
                fill.lineTo(first.x, first.y)

                val useBezier = valuesToDraw.size <= 220
                if (useBezier) {
                    for (i in 1 until valuesToDraw.size) {
                        val p0 = point((i - 2).coerceAtLeast(0))
                        val p1 = point(i - 1)
                        val p2 = point(i)
                        val p3 = point((i + 1).coerceAtMost(valuesToDraw.lastIndex))
                        val (c1, c2) = catmullRomCP(p0, p1, p2, p3)
                        path.cubicTo(c1.x, c1.y, c2.x, c2.y, p2.x, p2.y)
                        fill.cubicTo(c1.x, c1.y, c2.x, c2.y, p2.x, p2.y)
                    }
                } else {
                    for (i in 1 until valuesToDraw.size) {
                        val p = point(i)
                        path.lineTo(p.x, p.y)
                        fill.lineTo(p.x, p.y)
                    }
                }

                val last = point(valuesToDraw.lastIndex)
                fill.lineTo(last.x, topPad + chartHeight)
                fill.close()

                // Gradient fill
                drawPath(
                    path = fill,
                    brush = Brush.verticalGradient(
                        colors = listOf(
                            s.color.copy(alpha = 0.22f),
                            s.color.copy(alpha = 0.0f)
                        ),
                        startY = topPad,
                        endY   = topPad + chartHeight
                    )
                )

                // Stroke line
                drawPath(
                    path  = path,
                    color = s.color,
                    style = Stroke(width = 3.5f, cap = StrokeCap.Round)
                )

                // End-point glow dot
                drawCircle(color = s.color.copy(alpha = 0.4f), radius = 7f,  center = last)
                drawCircle(color = s.color,                    radius = 4f,  center = last)
            }

            // ── Step selector vertical line + data dots ───────────────────
            if (selectedGlobalIndex in startIndex until endIndex) {
                val local = selectedGlobalIndex - startIndex
                val span  = (endIndex - startIndex - 1).coerceAtLeast(1)
                val x     = leftPad + chartWidth * local / span.toFloat()
                drawLine(
                    color       = onSurfaceColor.copy(alpha = 0.35f),
                    start       = Offset(x, topPad),
                    end         = Offset(x, topPad + chartHeight),
                    strokeWidth = 1.5f
                )
                visibleSeries.forEach { s ->
                    val safeLocal = local.coerceIn(0, s.values.lastIndex)
                    if (s.values.isEmpty()) return@forEach
                    val normalized = ((s.values[safeLocal] - minValue) / range).toFloat()
                    val y = topPad + chartHeight * (1f - normalized)
                    drawCircle(color = Color.White.copy(alpha = 0.9f), radius = 7f, center = Offset(x, y))
                    drawCircle(color = s.color, radius = 5f, center = Offset(x, y))
                }
            }
        }

        // ── Footer: zoom info + selected step tooltip ─────────────────────
        val zoomLabel     = "Zoom ${String.format(Locale.US, "%.1f", zoomX)}x"
        val rangeLabel    = if (maxLen > 0) "s$startIndex–s$endIndex / $maxLen" else "No data"
        val selectedLabel = if (selectedGlobalIndex >= 0) {
            val values = chartSeries.mapNotNull { s ->
                if (selectedGlobalIndex in s.values.indices)
                    "${s.name} ${valueFormatter(s.values[selectedGlobalIndex])}"
                else null
            }
            if (values.isNotEmpty()) "Step $selectedGlobalIndex • ${values.joinToString(" · ")}"
            else "Step $selectedGlobalIndex"
        } else {
            "Tap to inspect  •  Drag horizontally to pan  •  Double-tap to zoom/reset"
        }

        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween
        ) {
            Text(zoomLabel,  style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
            Text(rangeLabel, style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
        }
        Text(
            text  = selectedLabel,
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )
    }
}

private fun downsampleEvenly(values: List<Double>, maxPoints: Int): List<Double> {
    if (values.size <= maxPoints || maxPoints < 3) return values

    val sampled = ArrayList<Double>(maxPoints)
    val lastIndex = values.lastIndex
    for (i in 0 until maxPoints) {
        val srcIdx = ((i.toDouble() / (maxPoints - 1).toDouble()) * lastIndex.toDouble())
            .roundToInt()
            .coerceIn(0, lastIndex)
        sampled.add(values[srcIdx])
    }
    return sampled
}
