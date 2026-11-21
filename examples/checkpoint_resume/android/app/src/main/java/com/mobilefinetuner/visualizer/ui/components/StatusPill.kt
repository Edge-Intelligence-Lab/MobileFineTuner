package com.mobilefinetuner.visualizer.ui.components

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import com.mobilefinetuner.visualizer.model.RunStatus

@Composable
fun StatusPill(status: RunStatus, modifier: Modifier = Modifier) {
    val (label, color) = when (status) {
        RunStatus.IDLE -> "Idle" to Color(0xFF9A8F80)
        RunStatus.RUNNING -> "Running" to Color(0xFF2A8D60)
        RunStatus.COMPLETED -> "Completed" to Color(0xFF2A6CC2)
        RunStatus.FAILED -> "Failed" to Color(0xFFC23831)
    }

    Row(
        modifier = modifier
            .clip(RoundedCornerShape(50))
            .background(color.copy(alpha = 0.15f))
            .padding(horizontal = 10.dp, vertical = 6.dp),
        horizontalArrangement = Arrangement.spacedBy(8.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        androidx.compose.foundation.layout.Box(
            modifier = Modifier
                .clip(CircleShape)
                .background(color)
                .padding(4.dp)
        )
        Text(
            text = label,
            style = MaterialTheme.typography.labelLarge,
            color = color
        )
    }
}
