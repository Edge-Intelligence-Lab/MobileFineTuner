package com.mobilefinetuner.visualizer.continuity

import android.content.Context
import androidx.test.core.app.ApplicationProvider
import androidx.test.platform.app.InstrumentationRegistry
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Test
import java.io.File

/** Explicit opt-in full-model Runner test; staging is not a system-picker test. */
class FullProfileCheck {
    @Test fun fullProfileCompletesAcrossFiniteSlices() {
        requireExplicitFixture(this)
        require(android.os.Build.HARDWARE in listOf("ranchu", "goldfish"))
        val context = ApplicationProvider.getApplicationContext<Context>()
        val store = TaskStore(context)
        val id = InstrumentationRegistry.getArguments().getString("profileTask") ?: "profile-qwen"
        require(!File(store.task(id), "run").exists()) { "Use a new task; never overwrite evidence" }
        val spec = store.spec(id)
        require(spec.getInt("steps") == 5 && spec.getInt("seq_len") == 64)
        require(spec.getInt("slice_seconds") == 120 && spec.getInt("pause_grace_seconds") == 300)
        require(File(store.task(id), "initial.safetensors").isFile)
        val attempts = JSONArray()
        try {
            store.allow(id, false, false)
            val runner = TrainingRunner(context)
            fun execute(stop: Int): TrainingRunner.Outcome {
                val started = android.os.SystemClock.elapsedRealtime()
                val outcome = runner.run(id, false, stopAfter = stop)
                attempts.put(JSONObject().put("outcome", outcome.toString())
                    .put("elapsed_ms", android.os.SystemClock.elapsedRealtime() - started)
                    .put("status", store.status(id)))
                assertTrue(store.status(id).toString(), outcome in listOf(
                    TrainingRunner.Outcome.COMPLETED, TrainingRunner.Outcome.SUSPENDED))
                return outcome
            }
            assertEquals(TrainingRunner.Outcome.SUSPENDED, execute(2))
            assertEquals(2, store.status(id).getInt("committed_updates"))
            var completed = false
            repeat(3) { if (!completed) completed = execute(0) == TrainingRunner.Outcome.COMPLETED }
            assertTrue("Five updates did not complete within bounded slices", completed)
            assertEquals(5, store.status(id).getInt("committed_updates"))
            val starts = File(store.task(id), "run/sessions").walkTopDown()
                .filter { it.name == "events.jsonl" }
                .map { JSONObject(it.readLines().first()).getInt("pid") }.toList()
            assertTrue(starts.size >= 2)
            assertEquals(starts.size, starts.distinct().size)
            TaskStore.write(File(store.task(id), "profile_result.json"), JSONObject()
                .put("passed", true).put("attempts", attempts).put("pids", JSONArray(starts))
                .put("scope", "Full Qwen seq64 native Runner; not SAF/UI/background job test"))
        } finally {
            store.pause(id)
            TaskStore.write(File(store.task(id), "profile_attempts.json"), JSONObject().put("attempts", attempts))
        }
    }
}
