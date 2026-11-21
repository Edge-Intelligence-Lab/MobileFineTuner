package com.mobilefinetuner.visualizer.continuity

import android.content.Context
import androidx.test.core.app.ApplicationProvider
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Test
import java.io.File
import java.util.UUID

/** Explicit opt-in: uses full Qwen assets staged by android_cold_start.py. */
class PretrainedParityCheck {
    @Test fun fullQwen10Equals4Plus6() {
        requireExplicitFixture(this)
        val context = ApplicationProvider.getApplicationContext<Context>()
        val store = TaskStore(context)
        val source = store.task("cold-start-qwen")
        require(File(source, "model/model.safetensors").isFile)
        fun create(): String {
            val id = UUID.randomUUID().toString()
            val dir = store.task(id).apply { mkdirs() }
            File(dir, "model").mkdirs()
            // Use the same copy semantics as SAF import; Android may deny hard links.
            File(source, "model/model.safetensors").copyTo(File(dir, "model/model.safetensors"))
            for (name in listOf("model/config.json", "train.jsonl", "valid.jsonl")) File(source, name).copyTo(File(dir, name))
            TaskStore.write(File(dir, "spec.json"), store.spec("cold-start-qwen")
                .put("steps", 10).put("seq_len", 4).put("slice_seconds", 120))
            store.allow(id, false, false)
            return id
        }
        fun state(id: String): Pair<JSONObject, File> {
            val run = File(store.task(id), "run")
            val current = JSONObject(File(run, "LATEST").readText()).getString("current")
            val dir = File(run, "training_state/$current")
            return JSONObject(File(dir, "state.json").readText()) to dir
        }
        val baseline = create()
        assertEquals(TrainingRunner.Outcome.COMPLETED, TrainingRunner(context).run(baseline, false))
        val resumed = create()
        File(store.task(baseline), "run/sessions").walkTopDown().single { it.name == "lora_initial.safetensors" }
            .copyTo(File(store.task(resumed), "initial.safetensors"))
        assertEquals(TrainingRunner.Outcome.SUSPENDED, TrainingRunner(context).run(resumed, false, stopAfter = 4))
        assertEquals(4, store.status(resumed).getInt("committed_updates"))
        assertEquals(TrainingRunner.Outcome.COMPLETED, TrainingRunner(context).run(resumed, false))
        val (a, ad) = state(baseline); val (b, bd) = state(resumed)
        assertArrayEquals(File(ad, "state.safetensors").readBytes(), File(bd, "state.safetensors").readBytes())
        assertEquals(a.getJSONObject("optimizer").toString(), b.getJSONObject("optimizer").toString())
        assertEquals(a.getString("dataset_state"), b.getString("dataset_state"))
        val ta = a.getJSONObject("trainer"); val tb = b.getJSONObject("trainer")
        assertArrayEquals(File(ta.getString("best_path")).readBytes(), File(tb.getString("best_path")).readBytes())
        ta.remove("best_path"); tb.remove("best_path"); ta.remove("timing"); tb.remove("timing")
        assertEquals(ta.toString(), tb.toString())
        fun updates(id: String) = File(store.task(id), "run/sessions").walkTopDown().filter { it.name == "events.jsonl" }
            .flatMap { it.readLines().map { line -> JSONObject(line) }.asSequence() }
            .filter { it.getString("type") == "update" }.sortedBy { it.getInt("completed_updates") }.toList()
        val au = updates(baseline); val bu = updates(resumed)
        assertEquals(10, au.size); assertEquals(10, bu.size)
        for ((x, y) in au.zip(bu)) for (key in listOf("loss", "lr", "grad_norm", "total_tokens", "ema_loss"))
            assertEquals(key, x.get(key), y.get(key))
        TaskStore.write(File(context.filesDir, "pretrained_parity_result.json"), JSONObject()
            .put("passed", true).put("baseline", baseline).put("resumed", resumed)
            .put("steps", 10).put("split", 4).put("seq", 4).put("batch", 1).put("rank", 2)
            .put("scope", "Full pretrained Qwen on Android arm64 emulator; not phone accuracy benchmark"))
    }
}
