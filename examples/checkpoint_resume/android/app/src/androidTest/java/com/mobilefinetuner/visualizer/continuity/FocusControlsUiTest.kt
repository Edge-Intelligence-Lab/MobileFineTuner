package com.mobilefinetuner.visualizer.continuity

import android.content.Context
import androidx.compose.ui.test.*
import androidx.compose.ui.test.junit4.createEmptyComposeRule
import androidx.test.core.app.ActivityScenario
import androidx.test.core.app.ApplicationProvider
import org.junit.Assert.*
import org.junit.Rule
import org.junit.Test

class FocusControlsUiTest {
    @get:Rule val ui = createEmptyComposeRule()
    @Test fun focusRequiresConfirmationAndDoesNotRestartWhenEnded() {
        val ctx = ApplicationProvider.getApplicationContext<Context>()
        val store = TaskStore(ctx)
        val id = store.createDemo()
        val focus = FocusWindow(ctx)
        try {
            ActivityScenario.launch<TrainingActivity>(TrainingService.openTaskIntent(ctx, id)).use {
                ui.waitUntil(10000) { ui.onAllNodes(hasText("Start / Resume") and isEnabled()).fetchSemanticsNodes().isNotEmpty() }
                val start = "Focus for one hour (release all training)"
                ui.onNodeWithText(start).performScrollTo().performClick()
                ui.onNodeWithText("Cancel").performClick()
                assertEquals(0L, focus.withGate { g -> g.remainingMs })
                ui.onNodeWithText(start).performScrollTo().performClick()
                ui.onNodeWithText("Start focus").performClick()
                ui.onNodeWithText("Start / Resume").assertIsNotEnabled()
                ui.onNodeWithText("End focus (", substring = true).performScrollTo().performClick()
                ui.onNodeWithText("Start / Resume").assertIsEnabled()
                assertFalse(store.control(id).getBoolean("enabled"))
                ui.onNodeWithText("Release resources now (replay unfinished update)").performScrollTo().performClick()
                ui.onNodeWithText("Cancel").performClick()
                assertFalse(store.control(id).has("release_through"))
            }
        } finally { focus.end(); store.task(id).deleteRecursively() }
    }
}
