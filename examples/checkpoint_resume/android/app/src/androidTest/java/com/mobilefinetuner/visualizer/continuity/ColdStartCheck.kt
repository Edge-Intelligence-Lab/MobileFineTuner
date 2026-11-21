package com.mobilefinetuner.visualizer.continuity

import android.content.Context
import androidx.test.core.app.ApplicationProvider
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Test
import java.io.File

/** Explicit opt-in integration test; requires adb-staged full Qwen weights. */
class ColdStartCheck {
    @Test fun loadingBudgetFailsOnceAndLongerSliceResumes() {
        requireExplicitFixture(this)
        val context = ApplicationProvider.getApplicationContext<Context>()
        val store = TaskStore(context)
        val id = "cold-start-qwen"
        val dir = store.task(id).apply { mkdirs() }
        require(File(dir, "model/model.safetensors").isFile) { "Stage the full model first" }
        val spec = JSONObject().put("model", "qwen").put("steps", 1).put("batch_size", 1)
            .put("seq_len", 8).put("lora_r", 2).put("grad_accum", 1).put("lr", 0.0002)
            .put("threads", 4).put("slice_seconds", 1).put("min_available_mb", 128)
        TaskStore.write(File(dir, "spec.json"), spec)
        store.allow(id, true, false)
        assertEquals(TrainingRunner.Outcome.FAILED, TrainingRunner(context).run(id, true))
        assertTrue(store.status(id).getString("detail").startsWith("No update:"))
        assertEquals(0, store.status(id).getInt("committed_updates"))
        assertFalse(store.control(id).getBoolean("automatic"))
        assertEquals(TrainingRunner.Outcome.DISABLED, TrainingRunner(context).run(id, true))
        store.updateControl(id) { it.put("slice_seconds", 120) }
        store.allow(id, false, false)
        assertEquals(TrainingRunner.Outcome.COMPLETED, TrainingRunner(context).run(id, false))
        assertEquals(1, store.status(id).getInt("committed_updates"))
        TaskStore.write(File(dir, "cold_start_result.json"), JSONObject().put("passed", true)
            .put("short_slice_seconds", 1).put("recovered_slice_seconds", 120))
    }
}
