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

    private fun suspended(stopAfter: Int = 4): String {
        val id = store.createDemo()
        TaskStore.write(File(store.task(id), "spec.json"), store.spec(id).put("steps", 8))
        store.allow(id, false, false)
        assertEquals(TrainingRunner.Outcome.SUSPENDED, TrainingRunner(context).run(id, false, stopAfter = stopAfter))
        return id
    }

    private fun singleGeneration(): String {
        // Recreate LATEST as published at update zero, using the real initial commit.
        val id = suspended(stopAfter = 1)
        val pointer = latest(id)
        val first = JSONObject(pointer.readText()).getString("previous")
        val manifest = File(store.task(id), "run/training_state/$first/manifest.json")
        assertEquals("", JSONObject(manifest.readText()).getString("previous"))
        TaskStore.write(pointer, JSONObject().put("schema", 1).put("current", first).put("previous", ""))
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

    @Test fun onlyInvalidGenerationPermitsNonDestructiveNewTaskAfterNativeVerdict() {
        val id = singleGeneration()
        val pointer = latest(id).readText()
        val file = tensors(id, "current")
        corrupt(file)
        val bytes = file.readBytes()
        assertCannotReset(id)
        assertEquals(TrainingRunner.Outcome.FAILED, TrainingRunner(context).run(id, false))
        val reopened = TaskStore(context)
        assertTrue(reopened.status(id).getString("detail").contains("exit=65,"))
        assertTrue(reopened.status(id).getBoolean("retry_as_new"))
        val next = reopened.retryAsNew(id)
        assertNotEquals(id, next)
        assertFalse(File(reopened.task(next), "run").exists())
        assertFalse(reopened.control(next).optBoolean("enabled"))
        assertEquals(pointer, latest(id).readText())
        assertArrayEquals(bytes, file.readBytes())
    }

    private fun assertUnreadableCurrentDoesNotFallback(id: String) {
        val pointer = latest(id).readText()
        val file = tensors(id, "current")
        val mode = android.system.Os.stat(file.path).st_mode
        try {
            android.system.Os.chmod(file.path, 0)
            assertEquals(TrainingRunner.Outcome.FAILED, TrainingRunner(context).run(id, false))
            assertFalse(store.status(id).getString("detail").contains("exit=65,"))
            assertCannotReset(id)
            assertEquals(pointer, latest(id).readText())
        } finally { android.system.Os.chmod(file.path, mode) }
        store.allow(id, false, false)
        assertEquals(TrainingRunner.Outcome.COMPLETED, TrainingRunner(context).run(id, false))
        val resume = File(store.task(id), "run/sessions").walkTopDown().filter { it.name == "events.jsonl" }
            .flatMap { f -> f.readLines().asSequence().map { JSONObject(it) } }.single { it.optString("type") == "resume" }
        assertEquals(JSONObject(pointer).getString("current"), resume.getString("generation"))
        assertFalse(resume.getBoolean("fallback"))
    }

    @Test fun unreadableCurrentWithValidPreviousDoesNotFallback() =
        assertUnreadableCurrentDoesNotFallback(suspended())

    @Test fun unreadableSoleGenerationDoesNotAuthorizeReset() =
        assertUnreadableCurrentDoesNotFallback(singleGeneration())

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
