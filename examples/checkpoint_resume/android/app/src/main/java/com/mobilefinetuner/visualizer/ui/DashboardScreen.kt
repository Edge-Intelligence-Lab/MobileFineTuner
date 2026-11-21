package com.mobilefinetuner.visualizer.ui

import android.net.Uri
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.AnimatedContent
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.core.MutableTransitionState
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.slideInVertically
import androidx.compose.animation.slideOutVertically
import androidx.compose.animation.togetherWith
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.IntrinsicSize
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.offset
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.rounded.ExpandMore
import androidx.compose.material.icons.rounded.FolderOpen
import androidx.compose.material.icons.rounded.Refresh
import androidx.compose.material3.Button
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Surface
import androidx.compose.material3.Tab
import androidx.compose.material3.TabRow
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import com.mobilefinetuner.visualizer.model.DashboardTab
import com.mobilefinetuner.visualizer.model.EventType
import com.mobilefinetuner.visualizer.model.RunHandle
import com.mobilefinetuner.visualizer.model.RunSnapshot
import com.mobilefinetuner.visualizer.model.RunStatus
import com.mobilefinetuner.visualizer.model.StepMetric
import com.mobilefinetuner.visualizer.ui.components.InteractiveMetricLineChart
import com.mobilefinetuner.visualizer.ui.components.MemoryWaterfallChart
import com.mobilefinetuner.visualizer.ui.components.MetricCard
import com.mobilefinetuner.visualizer.ui.components.StatusPill
import com.mobilefinetuner.visualizer.ui.components.TerminalLogView
import com.mobilefinetuner.visualizer.ui.theme.LossColor
import com.mobilefinetuner.visualizer.ui.theme.LrColor
import com.mobilefinetuner.visualizer.ui.theme.NeonAmber
import com.mobilefinetuner.visualizer.ui.theme.NeonBlue
import com.mobilefinetuner.visualizer.ui.theme.NeonGreen
import com.mobilefinetuner.visualizer.ui.theme.PplColor
import com.mobilefinetuner.visualizer.ui.theme.RssColor
import com.mobilefinetuner.visualizer.viewmodel.ExperimentViewModel
import kotlinx.coroutines.delay
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun DashboardScreen(
    viewModel: ExperimentViewModel,
    modifier: Modifier = Modifier
) {
    val ui by viewModel.uiState.collectAsState()

    val picker = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.OpenDocumentTree()
    ) { uri: Uri? ->
        if (uri != null) viewModel.onDirectoryPicked(uri)
    }

    Box(
        modifier = modifier
            .fillMaxSize()
            .background(
                brush = Brush.verticalGradient(
                    listOf(
                        Color(0xFFF2F2F7),
                        Color(0xFFFFFFFF),
                        Color(0xFFF2F8FF)
                    )
                )
            )
    ) {
        Box(
            modifier = Modifier
                .align(Alignment.TopEnd)
                .offset(x = 72.dp, y = (-40).dp)
                .size(240.dp)
                .clip(CircleShape)
                .background(Color(0x18007AFF))
        )
        Box(
            modifier = Modifier
                .align(Alignment.BottomStart)
                .offset(x = (-96).dp, y = 20.dp)
                .size(300.dp)
                .clip(CircleShape)
                .background(Color(0x1234C759))
        )

        Scaffold(
            modifier = Modifier.fillMaxSize(),
            containerColor = Color.Transparent,
            topBar = {
                TopAppBar(
                    colors = TopAppBarDefaults.topAppBarColors(containerColor = Color.Transparent),
                    title = {
                        Row(
                            verticalAlignment = Alignment.CenterVertically,
                            horizontalArrangement = Arrangement.spacedBy(10.dp)
                        ) {
                            // Brand accent gradient square
                            Box(
                                modifier = Modifier
                                    .size(9.dp)
                                    .background(
                                        brush = Brush.linearGradient(listOf(NeonBlue, NeonGreen)),
                                        shape = androidx.compose.foundation.shape.RoundedCornerShape(2.dp)
                                    )
                            )
                            Column {
                                Text(
                                    text  = "MobileFineTuner",
                                    style = MaterialTheme.typography.headlineSmall
                                )
                                Text(
                                    text     = ui.snapshot?.runName ?: "Ready to load runs",
                                    style    = MaterialTheme.typography.bodyMedium,
                                    color    = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.88f),
                                    maxLines = 1,
                                    overflow = TextOverflow.Ellipsis
                                )
                            }
                        }
                    },
                    actions = {
                        IconButton(onClick = { viewModel.refreshNow() }) {
                            Icon(Icons.Rounded.Refresh, contentDescription = "Refresh")
                        }
                    }
                )
            }
        ) { innerPadding ->
            Column(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(innerPadding)
            ) {
                ControlBar(
                    runs = ui.runHandles,
                    selectedRunId = ui.selectedRunId,
                    onPickDirectory = { picker.launch(null) },
                    onEnableDemo = viewModel::enableDemoMode,
                    onSelectRun = viewModel::selectRun
                )

                TabRow(
                    selectedTabIndex = ui.activeTab.ordinal,
                    modifier = Modifier
                        .padding(horizontal = 12.dp)
                        .clip(RoundedCornerShape(16.dp)),
                    containerColor = MaterialTheme.colorScheme.surface.copy(alpha = 0.84f)
                ) {
                    DashboardTab.entries.forEach { tab ->
                        Tab(
                            selected = tab == ui.activeTab,
                            onClick = { viewModel.selectTab(tab) },
                            text = {
                                Text(
                                    text = shortTabTitle(tab),
                                    maxLines = 1,
                                    overflow = TextOverflow.Clip,
                                    style = MaterialTheme.typography.bodyMedium
                                )
                            }
                        )
                    }
                }

                if (ui.error != null) {
                    Surface(
                        color = MaterialTheme.colorScheme.error.copy(alpha = 0.12f),
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(12.dp)
                    ) {
                        Text(
                            text = ui.error ?: "",
                            modifier = Modifier.padding(12.dp),
                            color = MaterialTheme.colorScheme.error
                        )
                    }
                }

                AnimatedContent(
                    targetState = ui.activeTab,
                    label = "tabSwitch",
                    transitionSpec = {
                        (fadeIn() + slideInVertically { it / 10 }) togetherWith
                            (fadeOut() + slideOutVertically { -it / 12 })
                    }
                ) { tab ->
                    when (tab) {
                        DashboardTab.OVERVIEW -> OverviewTab(snapshot = ui.snapshot)
                        DashboardTab.TRAINING -> TrainingTab(snapshot = ui.snapshot)
                        DashboardTab.LOGS -> LogsTab(snapshot = ui.snapshot)
                        DashboardTab.EXPERIMENTS -> ExperimentsTab(
                            runs = ui.runHandles,
                            selectedRunId = ui.selectedRunId,
                            onSelect = viewModel::selectRun
                        )
                    }
                }
            }
        }
    }
}

@Composable
private fun ControlBar(
    runs: List<RunHandle>,
    selectedRunId: String?,
    onPickDirectory: () -> Unit,
    onEnableDemo: () -> Unit,
    onSelectRun: (String) -> Unit
) {
    var primaryExpanded by remember { mutableStateOf(false) }
    val selected = runs.firstOrNull { it.id == selectedRunId }

    Column(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 12.dp, vertical = 6.dp),
        verticalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        // Action buttons row
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(10.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Button(onClick = onPickDirectory, modifier = Modifier.weight(1f)) {
                Icon(Icons.Rounded.FolderOpen, contentDescription = null)
                Text("Select Runs", modifier = Modifier.padding(start = 8.dp))
            }
            OutlinedButton(onClick = onEnableDemo) {
                Text("Load Demo")
            }
        }

        // Unified run selector — primary only
        Box(modifier = Modifier.fillMaxWidth()) {
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .clip(RoundedCornerShape(16.dp))
                    .border(1.dp, Color(0xFFDDE2EC), RoundedCornerShape(16.dp))
                    .background(Color(0xFFFAFBFF))
            ) {
                // Primary row
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .clickable(enabled = runs.isNotEmpty()) { primaryExpanded = true }
                        .padding(horizontal = 14.dp, vertical = 11.dp),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(10.dp)
                ) {
                    Text(
                        text  = "Primary",
                        style = MaterialTheme.typography.labelSmall,
                        color = Color(0xFF007AFF),
                        modifier = Modifier
                            .background(Color(0xFF007AFF).copy(alpha = 0.10f), RoundedCornerShape(6.dp))
                            .padding(horizontal = 7.dp, vertical = 3.dp)
                    )
                    Text(
                        text     = selected?.name ?: "Select a run",
                        style    = MaterialTheme.typography.bodyMedium,
                        color    = if (selected != null) MaterialTheme.colorScheme.onSurface
                                   else MaterialTheme.colorScheme.onSurfaceVariant,
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis,
                        modifier = Modifier.weight(1f)
                    )
                    Icon(
                        Icons.Rounded.ExpandMore,
                        contentDescription = null,
                        tint     = Color(0xFFB0B8C8),
                        modifier = Modifier.size(18.dp)
                    )
                }
            }

            // Primary dropdown
            DropdownMenu(expanded = primaryExpanded, onDismissRequest = { primaryExpanded = false }) {
                runs.take(60).forEach { run ->
                    DropdownMenuItem(
                        text = {
                            Text(run.name, maxLines = 1, overflow = TextOverflow.Ellipsis,
                                modifier = Modifier.width(280.dp))
                        },
                        onClick = { primaryExpanded = false; onSelectRun(run.id) }
                    )
                }
            }
        }
    }
}

@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun OverviewTab(snapshot: RunSnapshot?) {
    if (snapshot == null) {
        EmptyState("Select a run folder and a run to inspect training progress.")
        return
    }

    val last = snapshot.metrics.lastOrNull()
    val prev = snapshot.metrics.dropLast(1).lastOrNull()

    var pulse by remember(snapshot.runId) { mutableStateOf(false) }
    var seenStep by remember(snapshot.runId) { mutableIntStateOf(-1) }

    LaunchedEffect(last?.step) {
        val s = last?.step ?: return@LaunchedEffect
        if (s != seenStep) {
            seenStep = s
            pulse = true
            delay(1400)
            pulse = false
        }
    }

    val entryState = remember { MutableTransitionState(false).apply { targetState = true } }

    Box(modifier = Modifier.fillMaxSize().padding(top = 8.dp)) {
    LazyColumn(
        modifier = Modifier.fillMaxSize(),
        contentPadding = PaddingValues(12.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp)
    ) {
        item {
            AnimatedVisibility(
                visibleState = entryState,
                enter = fadeIn() + slideInVertically { it / 8 }
            ) {
                Card(
                    colors = CardDefaults.cardColors(
                        containerColor = MaterialTheme.colorScheme.surface.copy(alpha = 0.9f)
                    ),
                    elevation = CardDefaults.cardElevation(defaultElevation = 9.dp),
                    shape = RoundedCornerShape(22.dp),
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Column(
                        modifier = Modifier
                            .background(
                                brush = Brush.linearGradient(
                                    listOf(
                                        Color(0xFFEFF6FF),
                                        Color(0xFFFFFFFF),
                                        Color(0xFFF2F2F7)
                                    )
                                )
                            )
                            .padding(16.dp),
                        verticalArrangement = Arrangement.spacedBy(10.dp)
                    ) {
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceBetween,
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Text(
                                text = snapshot.runName,
                                style = MaterialTheme.typography.headlineSmall,
                                color = MaterialTheme.colorScheme.onSurface,
                                maxLines = 2,
                                overflow = TextOverflow.Ellipsis
                            )
                            StatusPill(snapshot.status)
                        }

                        Row(
                            horizontalArrangement = Arrangement.spacedBy(8.dp),
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Text(
                                text = if (snapshot.status == RunStatus.RUNNING) "Realtime stream" else "Snapshot mode",
                                style = MaterialTheme.typography.labelLarge,
                                color = MaterialTheme.colorScheme.primary
                            )
                            Text(
                                text = "•",
                                style = MaterialTheme.typography.labelLarge,
                                color = MaterialTheme.colorScheme.onSurfaceVariant
                            )
                            Text(
                                text = "${snapshot.metrics.size} steps parsed",
                                style = MaterialTheme.typography.labelLarge,
                                color = MaterialTheme.colorScheme.onSurfaceVariant
                            )
                        }

                        Text(
                            text = "Last update ${formatTime(snapshot.updatedAtMs)}",
                            style = MaterialTheme.typography.bodyMedium,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )

                        val total = snapshot.summary.totalSteps ?: last?.totalSteps
                        val stepText = if (total != null && last != null) {
                            "Step ${last.step} / $total"
                        } else if (last != null) {
                            "Step ${last.step}"
                        } else {
                            "No training step parsed"
                        }

                        val progress = if (total != null && total > 0 && last != null) {
                            (last.step.toFloat() / total.toFloat()).coerceIn(0f, 1f)
                        } else {
                            0f
                        }

                        // Circular progress ring
                        val primaryColor = MaterialTheme.colorScheme.primary
                        val surfaceColor = MaterialTheme.colorScheme.onSurface
                        Box(
                            modifier = Modifier.fillMaxWidth(),
                            contentAlignment = Alignment.Center
                        ) {
                            Canvas(modifier = Modifier.size(118.dp)) {
                                val sw = 13.dp.toPx()
                                // Track circle
                                drawArc(
                                    color      = surfaceColor.copy(alpha = 0.12f),
                                    startAngle = -90f,
                                    sweepAngle = 360f,
                                    useCenter  = false,
                                    style      = Stroke(width = sw, cap = StrokeCap.Round)
                                )
                                // Progress arc
                                if (progress > 0f) {
                                    drawArc(
                                        brush      = Brush.sweepGradient(
                                            listOf(
                                                primaryColor.copy(alpha = 0.5f),
                                                primaryColor
                                            )
                                        ),
                                        startAngle = -90f,
                                        sweepAngle = progress * 360f,
                                        useCenter  = false,
                                        style      = Stroke(width = sw, cap = StrokeCap.Round)
                                    )
                                }
                            }
                            Column(horizontalAlignment = Alignment.CenterHorizontally) {
                                Text(
                                    text  = "${(progress * 100).toInt()}%",
                                    style = MaterialTheme.typography.headlineSmall,
                                    color = MaterialTheme.colorScheme.primary,
                                    fontWeight = FontWeight.Bold
                                )
                                Spacer(Modifier.height(2.dp))
                                Text(
                                    text  = stepText,
                                    style = MaterialTheme.typography.bodyMedium,
                                    color = MaterialTheme.colorScheme.onSurfaceVariant
                                )
                            }
                        }
                    }
                }
            }
        }

        item {
            FlowRow(
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp),
                maxItemsInEachRow = 2
            ) {
                MetricCard(
                    label = "Loss",
                    value = last?.loss?.let { format3(it) } ?: "N/A",
                    deltaText = deltaText(last?.loss, prev?.loss, lowerIsBetter = true),
                    accent = LossColor,
                    modifier = Modifier.weight(1f),
                    isLiveHighlight = pulse,
                    sparklineValues = snapshot.metrics.takeLast(30).mapNotNull { it.loss }
                )
                MetricCard(
                    label = "PPL",
                    value = last?.ppl?.let { format2(it) } ?: "N/A",
                    deltaText = deltaText(last?.ppl, prev?.ppl, lowerIsBetter = true),
                    accent = PplColor,
                    modifier = Modifier.weight(1f),
                    isLiveHighlight = pulse,
                    sparklineValues = snapshot.metrics.takeLast(30).mapNotNull { it.ppl }
                )
                MetricCard(
                    label = "LR",
                    value = last?.lr?.let { format6(it) } ?: "N/A",
                    deltaText = deltaText(last?.lr, prev?.lr, lowerIsBetter = false),
                    accent = LrColor,
                    modifier = Modifier.weight(1f),
                    isLiveHighlight = pulse,
                    sparklineValues = snapshot.metrics.takeLast(30).mapNotNull { it.lr }
                )
                MetricCard(
                    label = "Peak RSS",
                    value = snapshot.summary.maxTrainRssMb?.let { "${format1(it)} MB" }
                        ?: snapshot.rssPoints.maxByOrNull { it.rssMb }?.rssMb?.let { "${format1(it)} MB" }
                        ?: "N/A",
                    accent = RssColor,
                    modifier = Modifier.weight(1f),
                    isLiveHighlight = pulse,
                    sparklineValues = snapshot.rssPoints.takeLast(30).map { it.rssMb }
                )
            }
        }

        item {
            MemoryWaterfallChart(
                modifier = Modifier.fillMaxWidth(),
                metrics = snapshot.metrics,
                rssPoints = snapshot.rssPoints
            )
        }

        item {
            EventTimeline(snapshot = snapshot)
        }
    }
    } // end Box
}

@Composable
private fun EventTimeline(snapshot: RunSnapshot) {
    val events = snapshot.events.takeLast(15).reversed()

    Column(
        modifier = Modifier
            .fillMaxWidth()
            .background(
                Brush.verticalGradient(
                    listOf(
                        MaterialTheme.colorScheme.surface.copy(alpha = 0.95f),
                        MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.88f)
                    )
                ),
                RoundedCornerShape(18.dp)
            )
            .padding(14.dp),
        verticalArrangement = Arrangement.spacedBy(0.dp)
    ) {
        Text(
            "Event Timeline",
            style = MaterialTheme.typography.titleLarge,
            color = MaterialTheme.colorScheme.onSurface
        )
        Spacer(Modifier.height(12.dp))

        if (events.isEmpty()) {
            Text(
                "No events parsed yet.",
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
        } else {
            events.forEachIndexed { idx, event ->
                val color   = eventTypeColor(event.type)
                val typeTag = event.type.name
                val isLast  = idx == events.lastIndex

                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(10.dp)
                ) {
                    // Left: dot + vertical stem
                    Column(
                        horizontalAlignment = Alignment.CenterHorizontally,
                        modifier = Modifier.width(12.dp)
                    ) {
                        Box(
                            modifier = Modifier
                                .size(10.dp)
                                .clip(CircleShape)
                                .background(color)
                        )
                        if (!isLast) {
                            Box(
                                modifier = Modifier
                                    .width(2.dp)
                                    .height(42.dp)
                                    .background(
                                        Brush.verticalGradient(
                                            listOf(
                                                color.copy(alpha = 0.45f),
                                                color.copy(alpha = 0.04f)
                                            )
                                        )
                                    )
                            )
                        }
                    }

                    // Right: type badge + step + message
                    Column(
                        verticalArrangement = Arrangement.spacedBy(3.dp),
                        modifier = Modifier.padding(bottom = if (isLast) 0.dp else 20.dp)
                    ) {
                        Row(
                            horizontalArrangement = Arrangement.spacedBy(8.dp),
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Text(
                                text     = typeTag,
                                style    = MaterialTheme.typography.labelLarge,
                                color    = color,
                                modifier = Modifier
                                    .background(color.copy(alpha = 0.15f), RoundedCornerShape(6.dp))
                                    .padding(horizontal = 6.dp, vertical = 2.dp)
                            )
                            if (event.step != null) {
                                Text(
                                    text  = "step ${event.step}",
                                    style = MaterialTheme.typography.labelLarge,
                                    color = MaterialTheme.colorScheme.onSurfaceVariant
                                )
                            }
                        }
                        Text(
                            text     = event.message,
                            style    = MaterialTheme.typography.bodyMedium,
                            color    = MaterialTheme.colorScheme.onSurface,
                            maxLines = 2,
                            overflow = TextOverflow.Ellipsis
                        )
                    }
                }
            }
        }
    }
}

@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun TrainingTab(snapshot: RunSnapshot?) {
    if (snapshot == null) {
        EmptyState("No run snapshot to draw chart.")
        return
    }

    val metrics = snapshot.metrics
    val last = metrics.lastOrNull()
    val lossSeries = metrics.mapNotNull { it.loss }
    val pplSeries = metrics.mapNotNull { it.ppl }
    val lrSeries = metrics.mapNotNull { it.lr }
    val rssSeries = snapshot.rssPoints.map { it.rssMb }
    val lossVarianceSeries = rollingLossDeltaVariance(lossSeries, window = 10)

    LazyColumn(
        modifier = Modifier.fillMaxSize(),
        contentPadding = PaddingValues(12.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp)
    ) {
        item {
            Card(
                modifier = Modifier.fillMaxWidth(),
                colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface.copy(alpha = 0.88f)),
                elevation = CardDefaults.cardElevation(defaultElevation = 8.dp),
                shape = RoundedCornerShape(20.dp)
            ) {
                Column(
                    modifier = Modifier.padding(14.dp),
                    verticalArrangement = Arrangement.spacedBy(10.dp)
                ) {
                    Text("Training Telemetry", style = MaterialTheme.typography.titleLarge)
                    Text(
                        "Live training visualization with zoom, pan, and step focus.",
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                    FlowRow(
                        horizontalArrangement = Arrangement.spacedBy(8.dp),
                        verticalArrangement = Arrangement.spacedBy(8.dp),
                        maxItemsInEachRow = 2
                    ) {
                        MiniMetricChip(label = "Loss", value = last?.loss?.let { format3(it) } ?: "N/A", accent = LossColor, modifier = Modifier.weight(1f))
                        MiniMetricChip(label = "PPL",  value = last?.ppl?.let { format2(it) } ?: "N/A",  accent = PplColor,  modifier = Modifier.weight(1f))
                        MiniMetricChip(label = "LR",   value = last?.lr?.let { format6(it) } ?: "N/A",   accent = LrColor,   modifier = Modifier.weight(1f))
                        MiniMetricChip(
                            label    = "RSS",
                            value    = rssSeries.lastOrNull()?.let { "${format1(it)} MB" } ?: "N/A",
                            accent   = RssColor,
                            modifier = Modifier.weight(1f)
                        )
                    }
                }
            }
        }

        item {
            InteractiveMetricLineChart(
                title          = "Loss",
                subtitle       = "Drag horizontally to pan, tap for step focus, double-tap zoom/reset",
                values         = lossSeries,
                color          = LossColor,
                valueFormatter = { format3(it) }
            )
        }
        item {
            InteractiveMetricLineChart(
                title          = "Loss Delta Variance (Proxy)",
                subtitle       = "Lower usually means more stable convergence",
                values         = lossVarianceSeries,
                color          = NeonAmber,
                seriesLabel    = "Variance",
                valueFormatter = { format4(it) }
            )
        }
        item {
            InteractiveMetricLineChart(
                title          = "Perplexity",
                subtitle       = "Lower is better — measures model uncertainty",
                values         = pplSeries,
                color          = PplColor,
                valueFormatter = { format2(it) }
            )
        }
        item {
            InteractiveMetricLineChart(
                title          = "Learning Rate",
                subtitle       = "Warmup + decay schedule",
                values         = lrSeries,
                color          = LrColor,
                valueFormatter = { format6(it) }
            )
        }
        item {
            InteractiveMetricLineChart(
                title          = "RSS Memory (MB)",
                subtitle       = "Peak resident set size over training",
                values         = rssSeries,
                color          = RssColor,
                valueFormatter = { format1(it) }
            )
        }
    }
}

@Composable
private fun LogsTab(snapshot: RunSnapshot?) {
    if (snapshot == null) {
        EmptyState("No logs loaded.")
        return
    }

    Column(modifier = Modifier.fillMaxSize().padding(12.dp)) {
        TerminalLogView(logs = snapshot.logTail, modifier = Modifier.fillMaxSize())
    }
}

@Composable
private fun MiniMetricChip(
    modifier: Modifier = Modifier,
    label: String,
    value: String,
    accent: Color = MaterialTheme.colorScheme.onSurfaceVariant
) {
    Column(
        modifier = modifier
            .heightIn(min = 72.dp)
            .background(accent.copy(alpha = 0.12f), RoundedCornerShape(14.dp))
            .padding(horizontal = 12.dp, vertical = 8.dp),
        verticalArrangement = Arrangement.spacedBy(3.dp)
    ) {
        Text(
            text  = label,
            style = MaterialTheme.typography.labelLarge,
            color = accent.copy(alpha = 0.80f)
        )
        Text(
            text  = value,
            style = MaterialTheme.typography.titleLarge,
            color = accent
        )
    }
}

@Composable
private fun ExperimentsTab(
    runs: List<RunHandle>,
    selectedRunId: String?,
    onSelect: (String) -> Unit
) {
    if (runs.isEmpty()) {
        EmptyState("No run discovered. Select your runs directory first.")
        return
    }

    LazyColumn(
        modifier = Modifier.fillMaxSize(),
        contentPadding = PaddingValues(12.dp),
        verticalArrangement = Arrangement.spacedBy(10.dp)
    ) {
        items(runs) { run ->
            val selected = run.id == selectedRunId

            val accentColor = if (selected) Color(0xFF007AFF) else Color(0xFFD1D1D6)
            val cardBg = if (selected) Color(0xFFEFF6FF) else Color(0xFFF8F9FB)
            val (badge, badgeColor) = inferModelBadge(run.name)
            val hasRss     = run.rssCsvUri     != null
            val hasMetrics = run.metricsNdjsonUri != null

            Card(
                colors    = CardDefaults.cardColors(containerColor = cardBg),
                elevation = CardDefaults.cardElevation(defaultElevation = 0.dp),
                shape     = RoundedCornerShape(18.dp)
            ) {
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(IntrinsicSize.Min)
                ) {
                    // ── Colored left accent strip ──────────────────────────
                    Box(
                        modifier = Modifier
                            .width(5.dp)
                            .fillMaxHeight()
                            .background(
                                accentColor,
                                RoundedCornerShape(topStart = 18.dp, bottomStart = 18.dp)
                            )
                    )

                    // ── Card content ───────────────────────────────────────
                    Column(
                        modifier = Modifier
                            .padding(start = 12.dp, top = 12.dp, end = 12.dp, bottom = 6.dp),
                        verticalArrangement = Arrangement.spacedBy(5.dp)
                    ) {
                        // Run name + model badge
                        Row(
                            horizontalArrangement = Arrangement.spacedBy(8.dp),
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Text(
                                text     = run.name,
                                style    = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.SemiBold),
                                maxLines = 1,
                                overflow = TextOverflow.Ellipsis,
                                modifier = Modifier.weight(1f, fill = false)
                            )
                            Text(
                                text     = badge,
                                style    = MaterialTheme.typography.labelMedium,
                                color    = badgeColor,
                                modifier = Modifier
                                    .background(badgeColor.copy(alpha = 0.14f), RoundedCornerShape(8.dp))
                                    .padding(horizontal = 8.dp, vertical = 3.dp)
                            )
                        }

                        // Timestamp + optional file-type pills
                        Row(
                            horizontalArrangement = Arrangement.spacedBy(6.dp),
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Text(
                                text  = "Updated ${formatTime(run.lastModifiedMs)}",
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant
                            )
                            if (hasRss) {
                                Text(
                                    text  = "RSS",
                                    style = MaterialTheme.typography.labelSmall,
                                    color = Color(0xFFAF52DE),
                                    modifier = Modifier
                                        .background(Color(0xFFAF52DE).copy(alpha = 0.10f), RoundedCornerShape(4.dp))
                                        .padding(horizontal = 5.dp, vertical = 2.dp)
                                )
                            }
                            if (hasMetrics) {
                                Text(
                                    text  = "NDJSON",
                                    style = MaterialTheme.typography.labelSmall,
                                    color = Color(0xFF34C759),
                                    modifier = Modifier
                                        .background(Color(0xFF34C759).copy(alpha = 0.10f), RoundedCornerShape(4.dp))
                                        .padding(horizontal = 5.dp, vertical = 2.dp)
                                )
                            }
                        }

                        // Action buttons row
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.Start
                        ) {
                            TextButton(
                                onClick  = { onSelect(run.id) },
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .background(
                                        if (selected) Color(0xFF007AFF).copy(alpha = 0.10f)
                                        else Color.Transparent,
                                        RoundedCornerShape(10.dp)
                                    )
                            ) {
                                Text(
                                    if (selected) "● Primary" else "Set Primary",
                                    color = Color(0xFF007AFF)
                                )
                            }
                        }
                    }
                }
            }
        }
    }
}

@Composable
private fun EmptyState(message: String) {
    Box(
        modifier = Modifier.fillMaxSize().padding(20.dp),
        contentAlignment = Alignment.Center
    ) {
        Text(
            text = message,
            style = MaterialTheme.typography.bodyLarge,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )
    }
}

private fun eventTypeColor(type: EventType): Color {
    return when (type) {
        EventType.INFO       -> Color(0xFF34C759)   // systemGreen
        EventType.EVAL       -> Color(0xFF007AFF)   // systemBlue
        EventType.CHECKPOINT -> Color(0xFFFF9500)   // systemOrange
        EventType.CLEANUP    -> Color(0xFFFF6B00)   // darker orange
        EventType.RSS        -> Color(0xFFAF52DE)   // systemPurple
        EventType.ENERGY     -> Color(0xFF5AC8FA)   // systemTeal
        EventType.WARNING    -> Color(0xFFFF9500)   // systemOrange
        EventType.ERROR      -> Color(0xFFFF3B30)   // systemRed
    }
}

private fun shortTabTitle(tab: DashboardTab): String {
    return when (tab) {
        DashboardTab.OVERVIEW -> "Home"
        DashboardTab.TRAINING -> "Train"
        DashboardTab.LOGS -> "Logs"
        DashboardTab.EXPERIMENTS -> "Runs"
    }
}

private fun formatTime(ms: Long): String {
    if (ms <= 0L) return "N/A"
    val fmt = SimpleDateFormat("MM-dd HH:mm:ss", Locale.getDefault())
    return fmt.format(Date(ms))
}

private fun format1(v: Double): String = String.format(Locale.US, "%.1f", v)
private fun format2(v: Double): String = String.format(Locale.US, "%.2f", v)
private fun format3(v: Double): String = String.format(Locale.US, "%.3f", v)
private fun format4(v: Double): String = String.format(Locale.US, "%.4f", v)
private fun format6(v: Double): String = String.format(Locale.US, "%.6f", v)

private fun deltaText(current: Double?, previous: Double?, lowerIsBetter: Boolean): String? {
    if (current == null || previous == null) return null
    val delta = current - previous
    val improved = if (lowerIsBetter) delta < 0 else delta > 0
    val arrow = if (improved) "▲" else "▼"
    val sign = if (delta >= 0) "+" else ""
    return "$arrow $sign${format4(delta)} vs prev"
}

private fun rollingLossDeltaVariance(lossSeries: List<Double>, window: Int): List<Double> {
    if (lossSeries.size < 3) return emptyList()
    val deltas = lossSeries.zipWithNext { a, b -> b - a }
    if (deltas.size < 2) return emptyList()
    val safeWindow = window.coerceAtLeast(2)
    return deltas.indices.map { idx ->
        val start = (idx - safeWindow + 1).coerceAtLeast(0)
        val slice = deltas.subList(start, idx + 1)
        val mean = slice.average()
        val variance = if (slice.size > 1) {
            slice.sumOf { d ->
                val diff = d - mean
                diff * diff
            } / (slice.size - 1).toDouble()
        } else {
            0.0
        }
        variance
    }
}

// ── Model type badge helper ────────────────────────────────────────────────────
private fun inferModelBadge(name: String): Pair<String, Color> {
    val lower = name.lowercase()
    return when {
        "gemma" in lower && "1b"     in lower -> "Gemma 1B"   to Color(0xFF34C759)   // systemGreen
        "gemma" in lower && "270"    in lower -> "Gemma 270M" to Color(0xFF34C759)
        "gemma" in lower                      -> "Gemma"       to Color(0xFF34C759)
        "gpt2"  in lower && "medium" in lower -> "GPT-2 M"    to Color(0xFF007AFF)   // systemBlue
        "gpt2"  in lower && "large"  in lower -> "GPT-2 L"    to Color(0xFF007AFF)
        "gpt2"  in lower                      -> "GPT-2"       to Color(0xFF007AFF)
        "lora"  in lower                      -> "LoRA"         to Color(0xFFAF52DE)   // systemPurple
        else                                  -> "Run"          to Color(0xFF8E8E93)   // systemGray
    }
}
