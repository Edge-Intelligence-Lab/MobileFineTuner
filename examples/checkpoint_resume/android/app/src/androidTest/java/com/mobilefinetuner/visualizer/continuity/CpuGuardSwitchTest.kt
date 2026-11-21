package com.mobilefinetuner.visualizer.continuity

import android.content.Context
import androidx.compose.ui.test.*
import androidx.compose.ui.test.junit4.createEmptyComposeRule
import androidx.test.core.app.ActivityScenario
import androidx.test.core.app.ApplicationProvider
import org.junit.Assert.*
import org.junit.Rule
import org.junit.Test
import java.io.File

class CpuGuardSwitchTest {
    @get:Rule val ui = createEmptyComposeRule()

    @Test fun runtimeSwitchSurvivesRelaunchWithoutEditingTheExperiment() {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val store = TaskStore(context)
        val id = store.createDemo()
        val other = store.createDemo()
        val spec = File(store.task(id), "spec.json")
        val original = spec.readBytes()
        val switch = hasText("Pause for CPU contention") and isToggleable() and isEnabled()
        fun open(task: String, check: () -> Unit) {
            ActivityScenario.launch<TrainingActivity>(TrainingService.openTaskIntent(context, task)).use {
                ui.waitUntil(10000) { ui.onAllNodes(switch).fetchSemanticsNodes().isNotEmpty() }
                check()
            }
        }
        try {
            open(id) {
                ui.onNode(switch).assertIsOff()
                val slice = ui.onAllNodes(hasSetTextAction())[0]
                slice.performScrollTo().performTextReplacement("47")
                ui.onNode(switch).performScrollTo().performClick()
                // Wait for an actual post-save status poll, not just the optimistic switch.
                ui.waitUntil(10000) {
                    ui.onAllNodesWithText("\"cpu_contention_guard\": true", substring = true)
                        .fetchSemanticsNodes().isNotEmpty()
                }
                slice.assertTextContains("47")
            }
            open(other) { ui.onNode(switch).assertIsOff() }
            open(id) { ui.onNode(switch).assertIsOn() }
            val control = TaskStore(context).control(id)
            assertTrue(TaskStore.cpuContentionEnabled(control, store.spec(id)))
            assertFalse(control.optBoolean("enabled"))
            assertFalse(control.has("slice_seconds"))
            assertFalse(store.hasAttemptArtifacts(id))
            assertArrayEquals(original, spec.readBytes())
        } finally {
            store.task(id).deleteRecursively()
            store.task(other).deleteRecursively()
        }
    }
}
