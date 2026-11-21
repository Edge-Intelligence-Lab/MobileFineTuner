package com.mobilefinetuner.visualizer.ui.components

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import java.util.Locale
import kotlin.math.max

@Composable
fun MetricLineChart(
    title: String,
    subtitle: String,
    values: List<Double>,
    color: Color,
    modifier: Modifier = Modifier,
    valueFormatter: (Double) -> String = { String.format(Locale.US, "%.3f", it) }
) {
    Column(
        modifier = modifier
            .background(
                brush = Brush.verticalGradient(
                    listOf(
                        color.copy(alpha = 0.08f),
                        MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.95f)
                    )
                ),
                shape = RoundedCornerShape(20.dp)
            )
            .padding(14.dp),
        verticalArrangement = Arrangement.spacedBy(10.dp)
    ) {
        Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
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
            Text(
                text = values.lastOrNull()?.let(valueFormatter) ?: "N/A",
                style = MaterialTheme.typography.titleLarge,
                color = color,
                textAlign = TextAlign.End
            )
        }

        Canvas(
            modifier = Modifier
                .fillMaxWidth()
                .height(180.dp)
        ) {
            if (values.isEmpty()) {
                drawRect(
                    color = color.copy(alpha = 0.06f),
                    size = size
                )
                return@Canvas
            }

            val maxValue = values.maxOrNull() ?: 1.0
            val minValue = values.minOrNull() ?: 0.0
            val range = max(1e-9, maxValue - minValue)

            val leftPad = 10f
            val rightPad = 10f
            val topPad = 12f
            val bottomPad = 22f
            val chartWidth = size.width - leftPad - rightPad
            val chartHeight = size.height - topPad - bottomPad

            for (i in 0..4) {
                val y = topPad + chartHeight * i / 4f
                drawLine(
                    color = Color.Gray.copy(alpha = 0.16f),
                    start = Offset(leftPad, y),
                    end = Offset(size.width - rightPad, y),
                    strokeWidth = 1f
                )
            }

            val linePath = Path()
            val fillPath = Path()

            fun point(index: Int): Offset {
                val x = leftPad + if (values.size == 1) {
                    chartWidth / 2f
                } else {
                    chartWidth * index / (values.size - 1).toFloat()
                }
                val normalized = ((values[index] - minValue) / range).toFloat()
                val y = topPad + chartHeight * (1f - normalized)
                return Offset(x, y)
            }

            val first = point(0)
            linePath.moveTo(first.x, first.y)
            fillPath.moveTo(first.x, topPad + chartHeight)
            fillPath.lineTo(first.x, first.y)

            for (i in 1 until values.size) {
                val p = point(i)
                linePath.lineTo(p.x, p.y)
                fillPath.lineTo(p.x, p.y)
            }

            val last = point(values.lastIndex)
            fillPath.lineTo(last.x, topPad + chartHeight)
            fillPath.close()

            drawPath(
                path = fillPath,
                brush = Brush.verticalGradient(
                    colors = listOf(
                        color.copy(alpha = 0.25f),
                        color.copy(alpha = 0.02f)
                    ),
                    startY = topPad,
                    endY = topPad + chartHeight
                )
            )

            drawPath(
                path = linePath,
                color = color,
                style = Stroke(width = 4f, cap = StrokeCap.Round)
            )

            drawCircle(
                color = color,
                radius = 5f,
                center = last
            )

            drawRect(
                color = color.copy(alpha = 0.08f),
                topLeft = Offset(0f, topPad + chartHeight),
                size = Size(size.width, bottomPad)
            )
        }

        Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
            Text(
                text = if (values.isNotEmpty()) "Start ${valueFormatter(values.first())}" else "Start",
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
            Text(
                text = "Points ${values.size}",
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
        }
    }
}
