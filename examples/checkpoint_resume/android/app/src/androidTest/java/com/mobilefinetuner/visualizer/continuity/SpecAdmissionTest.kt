package com.mobilefinetuner.visualizer.continuity

import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Test

class SpecAdmissionTest {
    private fun valid() = JSONObject().put("model", "qwen").put("steps", 10)
        .put("batch_size", 1).put("seq_len", 64).put("lora_r", 8)
        .put("grad_accum", 2).put("lr", 0.0002)

    private fun reject(key: String, value: Any) {
        val error = runCatching { TaskStore.validateSpec(valid().put(key, value)) }.exceptionOrNull()
        assertTrue("Accepted $key=$value", error is IllegalArgumentException)
    }

    @Test fun rejectsWrongIntegerTypesFractionsAndOverflow() {
        for (key in listOf("steps", "batch_size", "seq_len", "lora_r", "grad_accum",
                "min_available_mb", "slice_seconds", "threads", "shard_budget_mb", "pause_grace_seconds")) {
            for (value in listOf("oops", "2", true, JSONObject.NULL, 2.5, 2147483648L)) reject(key, value)
        }
        reject("lora_r", Int.MAX_VALUE)
        reject("seq_len", 1)
    }

    @Test fun rejectsWrongBooleanAndLearningRateTypes() {
        for (key in listOf("sharding", "activation_checkpointing", "memory_efficient_attention"))
            for (value in listOf(1, "true", "on", JSONObject.NULL)) reject(key, value)
        for (value in listOf("0.0002", "oops", true, JSONObject.NULL, 0.0, -0.1, 1e100, 1e-100)) reject("lr", value)
        reject("model", 1)
        reject("fixture", true)
    }

    @Test fun acceptsAbsentOptionsAndGenuineBooleansWithoutRewritingSpec() {
        TaskStore.validateSpec(valid())
        for (model in listOf("qwen", "gemma")) {
            for (enabled in listOf(true, false)) {
                val spec = valid().put("model", model).put("steps", 10.0)
                    .put("threads", 4).put("slice_seconds", 120).put("pause_grace_seconds", 420)
                    .put("sharding", enabled).put("activation_checkpointing", enabled)
                    .put("memory_efficient_attention", enabled).put("fixture", "test")
                val before = spec.toString()
                TaskStore.validateSpec(spec)
                assertEquals(before, spec.toString())
                assertEquals(10, spec.getInt("steps"))
                assertEquals(enabled, spec.getBoolean("sharding"))
            }
        }
    }
}
