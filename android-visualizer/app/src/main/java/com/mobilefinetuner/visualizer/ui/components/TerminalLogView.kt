package com.mobilefinetuner.visualizer.ui.components

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.itemsIndexed
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.derivedStateOf
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import com.mobilefinetuner.visualizer.ui.theme.MonoFontFamily

private data class LogChip(val label: String, val filter: String)

private val logChips = listOf(
    LogChip("All",    ""),
    LogChip("Train",  "[train]"),
    LogChip("Eval",   "[eval]"),
    LogChip("Ckpt",   "[checkpoint]"),
    LogChip("RSS",    "[rss"),
    LogChip("Energy", "[energy]"),
    LogChip("Warn",   "warn"),
    LogChip("Error",  "error")
)

// ── Terminal + chip palette (iOS-style chips, always-dark terminal) ────────────
private val TerminalBg     = Color(0xFF1C1C1E)   // iOS dark systemBackground
private val ChipSelected   = Color(0xFF007AFF)   // iOS systemBlue
private val ChipUnselected = Color(0xFFE5E5EA)   // iOS secondaryGrouped
private val ChipTextSel    = Color(0xFFFFFFFF)   // white on blue
private val ChipTextUnsel  = Color(0xFF6C6C70)   // iOS secondaryLabel

@Composable
fun TerminalLogView(
    logs: List<String>,
    modifier: Modifier = Modifier
) {
    var filter        by remember { mutableStateOf("") }
    var selectedChip  by remember { mutableStateOf("") }
    val chipListState = rememberLazyListState()

    // Chip takes priority over text filter
    val shown = when {
        selectedChip.isNotBlank() -> logs.filter { it.contains(selectedChip, ignoreCase = true) }
        filter.isNotBlank()       -> logs.filter { it.contains(filter, ignoreCase = true) }
        else                      -> logs
    }

    Column(
        modifier = modifier
            .background(
                Brush.verticalGradient(
                    listOf(
                        MaterialTheme.colorScheme.surface.copy(alpha = 0.9f),
                        MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.86f)
                    )
                ),
                RoundedCornerShape(18.dp)
            )
            .padding(12.dp),
        verticalArrangement = Arrangement.spacedBy(10.dp)
    ) {
        // ── Header ────────────────────────────────────────────────────────
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Text(
                text  = "Live Log",
                style = MaterialTheme.typography.titleLarge,
                color = MaterialTheme.colorScheme.onSurface
            )
            Text(
                text  = "${shown.size} / ${logs.size} lines",
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
        }

        // ── Quick filter chips (true horizontal drag + edge fade hints) ──
        val leftOverflow by remember {
            derivedStateOf {
                chipListState.firstVisibleItemIndex > 0 || chipListState.firstVisibleItemScrollOffset > 0
            }
        }
        val rightOverflow by remember {
            derivedStateOf {
                (chipListState.layoutInfo.visibleItemsInfo.lastOrNull()?.index ?: 0) < logChips.lastIndex
            }
        }

        Box(modifier = Modifier.fillMaxWidth()) {
            LazyRow(
                state = chipListState,
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(6.dp),
                contentPadding = PaddingValues(horizontal = 2.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                items(logChips.size) { idx ->
                    val chip = logChips[idx]
                    val count = if (chip.filter.isBlank()) logs.size
                    else logs.count { it.contains(chip.filter, ignoreCase = true) }
                    val isSelected = selectedChip == chip.filter
                    val bgColor = if (isSelected) ChipSelected else ChipUnselected
                    val textColor = if (isSelected) ChipTextSel else ChipTextUnsel

                    Text(
                        text = "${chip.label} ($count)",
                        style = MaterialTheme.typography.labelLarge,
                        color = textColor,
                        modifier = Modifier
                            .background(bgColor, RoundedCornerShape(20.dp))
                            .clickable {
                                selectedChip = if (isSelected) "" else chip.filter
                                if (!isSelected) filter = ""   // clear text field when chip selected
                            }
                            .padding(horizontal = 12.dp, vertical = 6.dp)
                    )
                }
            }

            if (leftOverflow) {
                Box(
                    modifier = Modifier
                        .align(Alignment.CenterStart)
                        .fillMaxWidth(0.07f)
                        .background(
                            Brush.horizontalGradient(
                                colors = listOf(
                                    MaterialTheme.colorScheme.surface.copy(alpha = 0.92f),
                                    Color.Transparent
                                )
                            )
                        )
                )
            }
            if (rightOverflow) {
                Box(
                    modifier = Modifier
                        .align(Alignment.CenterEnd)
                        .fillMaxWidth(0.07f)
                        .background(
                            Brush.horizontalGradient(
                                colors = listOf(
                                    Color.Transparent,
                                    MaterialTheme.colorScheme.surface.copy(alpha = 0.92f)
                                )
                            )
                        )
                )
            }
        }

        if (leftOverflow || rightOverflow) {
            Text(
                text = "Swipe left/right to view all filters",
                style = MaterialTheme.typography.labelMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
        }

        // ── Text filter field ─────────────────────────────────────────────
        OutlinedTextField(
            value         = filter,
            onValueChange = {
                filter       = it
                selectedChip = ""   // deselect chip when typing
            },
            label       = { Text("Filter") },
            placeholder = { Text("loss / eval / rss / warning …") },
            singleLine  = true,
            modifier    = Modifier.fillMaxWidth()
        )

        // ── Terminal output ───────────────────────────────────────────────
        LazyColumn(
            modifier = Modifier
                .fillMaxWidth()
                .background(TerminalBg, RoundedCornerShape(14.dp))
                .padding(vertical = 8.dp, horizontal = 10.dp),
            verticalArrangement = Arrangement.spacedBy(3.dp)
        ) {
            itemsIndexed(shown.takeLast(500)) { _, line ->
                Text(
                    text  = line,
                    style = MaterialTheme.typography.bodyMedium.copy(fontFamily = MonoFontFamily),
                    color = lineColor(line)
                )
            }
        }
    }
}

private fun lineColor(line: String): Color {
    val lower = line.lowercase()
    return when {
        "[train]"      in lower || "[step " in lower -> Color(0xFF8CE99A)
        "[eval]"       in lower                      -> Color(0xFF7ED6FF)
        "[checkpoint]" in lower                      -> Color(0xFFFACF6C)
        "[rss"         in lower || "memory" in lower -> Color(0xFFCDB5FF)
        "[energy]"     in lower || "sleep"  in lower -> Color(0xFFFFB870)
        "error"        in lower || "failed" in lower -> Color(0xFFFF8D8D)
        "warn"         in lower || "⚠"    in line  -> Color(0xFFFFD37D)
        else                                         -> Color(0xFFE6EDF3)
    }
}
