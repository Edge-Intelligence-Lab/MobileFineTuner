package com.mobilefinetuner.visualizer.model

import android.net.Uri

enum class RunStatus {
    IDLE,
    RUNNING,
    COMPLETED,
    FAILED
}

enum class EventType {
    INFO,
    EVAL,
    CHECKPOINT,
    CLEANUP,
    RSS,
    ENERGY,
    WARNING,
    ERROR
}

enum class DashboardTab(val title: String) {
    OVERVIEW("Overview"),
    TRAINING("Training"),
    COMPARISON("Comparison"),
    LOGS("Logs"),
    EXPERIMENTS("Experiments")
}

data class StepMetric(
    val step: Int,
    val totalSteps: Int? = null,
    val loss: Double? = null,
    val ppl: Double? = null,
    val lr: Double? = null,
    val tokens: Int? = null,
    val source: String = "log"
)

data class RssPoint(
    val index: Int,
    val rssMb: Double,
    val timeLabel: String? = null
)

data class RunEvent(
    val step: Int? = null,
    val type: EventType,
    val message: String,
    val raw: String
)

data class RunSummary(
    val maxInitRssMb: Double? = null,
    val maxTrainRssMb: Double? = null,
    val maxStepEndRssMb: Double? = null,
    val finalEmaLoss: Double? = null,
    val totalTokens: Long? = null,
    val totalSteps: Int? = null
)

data class RunSnapshot(
    val runId: String,
    val runName: String,
    val status: RunStatus,
    val metrics: List<StepMetric>,
    val rssPoints: List<RssPoint>,
    val events: List<RunEvent>,
    val logTail: List<String>,
    val summary: RunSummary,
    val updatedAtMs: Long
)

data class RunHandle(
    val id: String,
    val name: String,
    val dirUri: Uri,
    val trainLogUri: Uri,
    val rssCsvUri: Uri? = null,
    val metricsNdjsonUri: Uri? = null,
    val lastModifiedMs: Long
)

data class DashboardUiState(
    val selectedRootUri: Uri? = null,
    val activeTab: DashboardTab = DashboardTab.OVERVIEW,
    val runHandles: List<RunHandle> = emptyList(),
    val selectedRunId: String? = null,
    val compareRunId: String? = null,
    val snapshot: RunSnapshot? = null,
    val compareSnapshot: RunSnapshot? = null,
    val isLoading: Boolean = false,
    val error: String? = null
)
