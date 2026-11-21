package com.mobilefinetuner.visualizer.continuity

import android.content.Context
import androidx.test.core.app.ApplicationProvider
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Test
import java.io.File
import java.io.RandomAccessFile

class NativeCheckpointRecoveryTest {
    private val context = ApplicationProvider.getApplicationContext<Context>()
    private val store = TaskStore(context)

    private fun suspended(): String {
        val id = store.createDemo()
        TaskStore.write(File(store.task(id), "spec.json"), store.spec(id).put("steps", 8))
        store.allow(id, false, false)
        assertEquals(TrainingRunner.Outcome.SUSPENDED, TrainingRunner(context).run(id, false, stopAfter = 4))
        return id
    }

    private fun latest(id: String) = File(store.task(id), "run/LATEST")
    private fun tensors(id: String, slot: String): File {
        val name = JSONObject(latest(id).readText()).getString(slot)
        return File(store.task(id), "run/training_state/$name/state.safetensors")
    }
    private fun corrupt(file: File) {
        // Impossible header length. File size, manifests and cheap Android checks still match.
        RandomAccessFile(file, "rw").use { it.writeLong(-1L) }
    }
    private fun assertCannotReset(id: String) {
        assertFalse(store.status(id).getBoolean("retry_as_new"))
        assertTrue(runCatching { store.retryAsNew(id) }.exceptionOrNull() is IllegalStateException)
    }

    @Test fun bothInvalidPermitOnlyNonDestructiveNewTaskAndStaleFailureCannotReauthorize() {
        val id = suspended()
        val oldGeneration = store.control(id).getLong("resume_generation")
        val pointer = latest(id).readText()
        val files = listOf(tensors(id, "current"), tensors(id, "previous"))
        files.forEach { corrupt(it) }
        val payloads = files.map { it.readBytes() }
        assertTrue(store.status(id).getBoolean("checkpoint_available"))
        assertCannotReset(id)
        assertEquals(TrainingRunner.Outcome.FAILED, TrainingRunner(context).run(id, false))
        val reopened = TaskStore(context)
        assertTrue(reopened.status(id).getBoolean("retry_as_new"))
        assertFalse(reopened.status(id).getBoolean("checkpoint_available"))
        val next = reopened.retryAsNew(id)
        assertNotEquals(id, next)
        assertFalse(File(reopened.task(next), "run").exists())
        assertFalse(reopened.control(next).optBoolean("enabled"))
        assertEquals(pointer, latest(id).readText())
        files.forEachIndexed { i, f -> assertArrayEquals(payloads[i], f.readBytes()) }
        reopened.allow(id, false, false)
        assertFalse(reopened.fail(id, "late validation", oldGeneration, exhaustedPointer = pointer))
        assertCannotReset(id)
    }

    @Test fun invalidCurrentResumesTheRecordedPredecessor() {
        val id = suspended()
        val previous = JSONObject(latest(id).readText()).getString("previous")
        corrupt(tensors(id, "current"))
        assertEquals(TrainingRunner.Outcome.COMPLETED, TrainingRunner(context).run(id, false))
        val event = File(store.task(id), "run/sessions").walkTopDown().filter { it.name == "events.jsonl" }
            .flatMap { f -> f.readLines().asSequence().map { JSONObject(it) } }.single { it.optString("type") == "resume" }
        assertTrue(event.getBoolean("fallback"))
        assertEquals(previous, event.getString("generation"))
        assertCannotReset(id)
    }

    @Test fun temporarilyUnavailableCandidateFilesDoNotAuthorizeReset() {
        val id = suspended()
        val files = listOf(tensors(id, "current"), tensors(id, "previous"))
        val moved = mutableListOf<Pair<File, File>>()
        try {
            for (f in files) {
                val hidden = File(f.parentFile, "temporarily-offline")
                assertTrue(f.renameTo(hidden))
                moved.add(hidden to f)
            }
            assertNull(store.checkpoint(id))
            assertEquals(TrainingRunner.Outcome.FAILED, TrainingRunner(context).run(id, false))
            assertCannotReset(id)
        } finally { moved.forEach { (hidden, f) -> assertTrue(hidden.renameTo(f)) } }
        store.allow(id, false, false)
        assertEquals(TrainingRunner.Outcome.COMPLETED, TrainingRunner(context).run(id, false))
        assertCannotReset(id)
    }
}
