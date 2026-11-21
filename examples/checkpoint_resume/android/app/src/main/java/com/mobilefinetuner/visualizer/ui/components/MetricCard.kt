package com.mobilefinetuner.visualizer.ui.components

import androidx.compose.animation.core.FastOutSlowInEasing
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import kotlin.math.max

@Composable
fun MetricCard(
    label: String,
    value: String,
    accent: Color,
    modifier: Modifier = Modifier,
    deltaText: String? = null,
    isLiveHighlight: Boolean = false,
    sparklineValues: List<Double> = emptyList()
) {
    val pulse = if (isLiveHighlight) {
        rememberInfiniteTransition(label = "metricPulse").animateFloat(
            initialValue = 0.2f,
            targetValue  = 0.85f,
            animationSpec = infiniteRepeatable(
                animation  = tween(durationMillis = 900, easing = FastOutSlowInEasing),
                repeatMode = RepeatMode.Reverse
            ),
            label = "metricPulseAnim"
        ).value
    } else {
        0.2f
    }

    val borderAlpha = if (isLiveHighlight) 0.55f * pulse else 0.22f

    Column(
        modifier = modifier
            .border(
                width  = 1.dp,
                color  = accent.copy(alpha = borderAlpha),
                shape  = RoundedCornerShape(18.dp)
            )
            .background(
                brush = Brush.linearGradient(
                    listOf(
                        accent.copy(alpha = 0.14f + pulse * 0.10f),
                        MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.95f)
                    )
                ),
                shape = RoundedCornerShape(18.dp)
            )
            .heightIn(min = 120.dp)
            .padding(horizontal = 14.dp, vertical = 12.dp),
        verticalArrangement = Arrangement.spacedBy(5.dp)
    ) {
        Text(
            text  = label,
            style = MaterialTheme.typography.labelLarge,
            color = accent.copy(alpha = 0.85f)
        )
        Text(
            text       = value,
            style      = MaterialTheme.typography.titleLarge,
            color      = MaterialTheme.colorScheme.onSurface,
            fontWeight = FontWeight.Bold
        )
        Row {
            Text(
                text  = deltaText ?: " ",
                style = MaterialTheme.typography.bodyMedium,
                color = if (deltaText.isNullOrBlank()) Color.Transparent else accent
            )
        }

        // ── Sparkline micro-chart ─────────────────────────────────────────
        if (sparklineValues.size >= 3) {
            Canvas(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(top = 2.dp)
                    .heightIn(min = 30.dp, max = 34.dp)
            ) {
                val maxV  = sparklineValues.maxOrNull() ?: 1.0
                val minV  = sparklineValues.minOrNull() ?: 0.0
                val range = max(1e-9, maxV - minV)
                val w     = size.width
                val h     = size.height

                val path = Path()
                val fill = Path()

                sparklineValues.forEachIndexed { i, v ->
                    val x          = w * i / (sparklineValues.size - 1).toFloat()
                    val normalized = ((v - minV) / range).toFloat()
                    val y          = h * (1f - normalized)
                    if (i == 0) {
                        path.moveTo(x, y)
                        fill.moveTo(x, h)
                        fill.lineTo(x, y)
                    } else {
                        // Simple bezier for sparkline smoothness
                        val prev = sparklineValues[i - 1]
                        val prevX = w * (i - 1) / (sparklineValues.size - 1).toFloat()
                        val prevY = h * (1f - ((prev - minV) / range).toFloat())
                        val cpX = (prevX + x) / 2f
                        path.cubicTo(cpX, prevY, cpX, y, x, y)
                        fill.cubicTo(cpX, prevY, cpX, y, x, y)
                    }
                }

                val lastX = w
                val lastV = sparklineValues.last()
                val lastY = h * (1f - ((lastV - minV) / range).toFloat())
                fill.lineTo(lastX, h)
                fill.close()

                // Gradient fill under sparkline
                drawPath(
                    path  = fill,
                    brush = Brush.verticalGradient(
                        colors = listOf(accent.copy(alpha = 0.25f), Color.Transparent),
                        startY = 0f,
                        endY   = h
                    )
                )

                // Sparkline stroke
                drawPath(
                    path  = path,
                    color = accent.copy(alpha = 0.80f),
                    style = Stroke(width = 2f, cap = StrokeCap.Round)
                )

                // Endpoint dot
                drawCircle(
                    color  = accent,
                    radius = 3.5f,
                    center = androidx.compose.ui.geometry.Offset(lastX, lastY)
                )
            }
        }
    }
}
