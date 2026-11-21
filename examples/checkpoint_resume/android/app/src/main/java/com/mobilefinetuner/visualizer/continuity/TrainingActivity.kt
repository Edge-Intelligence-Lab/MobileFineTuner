package com.mobilefinetuner.visualizer.continuity

import android.Manifest
import android.content.Intent
import android.os.Build
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.selection.toggleable
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.ui.Modifier
import androidx.compose.ui.Alignment
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import com.mobilefinetuner.visualizer.MainActivity
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.async
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class TrainingActivity : ComponentActivity() {
    companion object {
        // Only the bounded admission + service handoff; training remains in the service/worker.
        private val admissions = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    }
    private data class RuntimeControls(val automatic: Boolean, val charging: Boolean, val slice: Int, val grace: Int)
    private fun pauseTask(id: String, immediately: Boolean = false) {
        val app = applicationContext
        val generation = TaskStore(app).pause(id, immediately)
        admissions.launch {
            try { Scheduling.cancel(app, id, generation) }
            catch (e: Exception) { android.util.Log.w("TrainingActivity", "Paused; background work cleanup failed", e) }
        }
    }
    private var taskIntent by mutableStateOf<Intent?>(null)
    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        setIntent(intent)
        taskIntent = intent
    }
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        taskIntent = if (savedInstanceState == null) intent else null
        setContent {
            val store = remember { TaskStore(this) }
            val focus = remember { FocusWindow(applicationContext) }
            var focusMs by remember { mutableLongStateOf(0L) }
            var confirmFocus by rememberSaveable { mutableStateOf(false) }
            var confirmRelease by rememberSaveable { mutableStateOf<String?>(null) }
            var ids by remember { mutableStateOf(store.list()) }
            var selected by rememberSaveable { mutableStateOf<String?>(null) }
            var progressLine by remember(selected) { mutableStateOf("") }
            var status by remember { mutableStateOf("") }
            var pressureMessage by remember(selected) { mutableStateOf("") }
            var message by remember { mutableStateOf("") }
            var automatic by remember { mutableStateOf(false) }
            var charging by remember { mutableStateOf(true) }
            var cpuGuard by remember(selected) { mutableStateOf(false) }
            var cpuSaving by remember(selected) { mutableStateOf(false) }
            var cpuEditVersion by remember(selected) { mutableIntStateOf(0) }
            var importing by remember { mutableStateOf(false) }
            var menu by remember { mutableStateOf(false) }
            var sliceText by remember { mutableStateOf("30") }
            var graceText by remember { mutableStateOf("30") }
            var starting by remember { mutableStateOf(false) }
            var retryAsNew by remember { mutableStateOf(false) }
            var showBatch by remember(selected) { mutableStateOf(false) }
            var originalBatch by remember(selected) { mutableIntStateOf(1) }
            var committedBatch by remember(selected) { mutableIntStateOf(1) }
            var previousBatch by remember(selected) { mutableIntStateOf(1) }
            var effectiveBatch by remember(selected) { mutableLongStateOf(1L) }
            var batchChoice by rememberSaveable(selected) { mutableStateOf<Int?>(null) }
            var batchMenu by remember(selected) { mutableStateOf(false) }
            var loadedTask by remember { mutableStateOf<String?>(null) }
            val scope = rememberCoroutineScope()
            LaunchedEffect(Unit) {
                while (true) {
                    focusMs = withContext(Dispatchers.IO) { focus.withGate { it.remainingMs } }
                    delay(1000)
                }
            }
            LaunchedEffect(taskIntent) {
                ids = withContext(Dispatchers.IO) { store.list() }
                val requested = taskIntent?.getStringExtra("task")
                if (taskIntent != null) {
                    selected = if (requested != null) ids.firstOrNull { it == requested } else selected ?: store.lastSelected(ids)
                } else if (selected != null && selected !in ids) selected = null
                if (requested != null && selected == null) message = "Requested task is unavailable; select an existing task."
            }
            val exporter = rememberLauncherForActivityResult(ActivityResultContracts.CreateDocument("application/zip")) { uri ->
                val id = selected
                if (uri != null && id != null) scope.launch {
                    try {
                        withContext(Dispatchers.IO) { store.exportRecords(id, uri) }
                        message = "Exported records and checkpoints (no frozen model). Resume stays in this app's original task."
                    } catch (e: Exception) { message = e.message ?: "Export failed" }
                }
            }
            val permissions = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { }
            val importer = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocumentTree()) { uri ->
                if (uri != null) scope.launch {
                    importing = true
                    try {
                        selected = withContext(Dispatchers.IO) { store.importTree(uri) }
                        ids = store.list(); message = "Imported; assets remain in private local storage."
                    } catch (e: Exception) { message = e.message ?: "Import failed" }
                    finally { importing = false }
                }
            }
            LaunchedEffect(selected) {
                loadedTask = null
                retryAsNew = false
                val id = selected
                if (id != null) store.rememberSelected(id)
                var persistedControls: RuntimeControls? = null
                while (true) {
                    if (id == null) { status = "No task"; break }
                    try {
                        val readVersion = cpuEditVersion
                        val snapshot = withContext(Dispatchers.IO) {
                            store.respectSystemStop(id)
                            store.status(id) to store.spec(id)
                        }
                        val control = snapshot.first.getJSONObject("control")
                        if (!cpuSaving && readVersion == cpuEditVersion)
                            cpuGuard = TaskStore.cpuContentionEnabled(control, snapshot.second)
                        originalBatch = snapshot.second.getInt("batch_size")
                        committedBatch = snapshot.first.optInt("committed_physical_batch", originalBatch)
                        previousBatch = control.optInt("resume_microbatch", committedBatch)
                        effectiveBatch = originalBatch.toLong() * snapshot.second.getInt("grad_accum")
                        showBatch = snapshot.first.optBoolean("checkpoint_available") &&
                            !snapshot.first.optBoolean("execution_active") &&
                            snapshot.first.optString("committed_status") != "completed"
                        val slice = control.optInt("slice_seconds", snapshot.second.optInt("slice_seconds", 30))
                        val grace = control.optInt("pause_grace_seconds", snapshot.second.optInt("pause_grace_seconds", 30))
                        val controls = RuntimeControls(control.optBoolean("automatic"), control.optBoolean("charging", true), slice, grace)
                        // Refresh external changes without overwriting unsaved edits on every poll.
                        if (controls != persistedControls) {
                            automatic = control.optBoolean("automatic")
                            charging = control.optBoolean("charging", true)
                            sliceText = slice.toString()
                            graceText = grace.toString()
                            persistedControls = controls
                        }
                        status = snapshot.first.toString(2)
                        val live = snapshot.first.optBoolean("execution_active")
                        val committed = snapshot.first.optInt("committed_updates", -1)
                        val phase = if (!live) "Trainer not running" else if (!control.optBoolean("enabled")) "Stopping trainer" else "Trainer active"
                        progressLine = if (committed < 0) "$phase · No committed update yet"
                            else "$phase · Last checkpoint: update $committed"
                        pressureMessage = when {
                            control.has("cpu_hold_reason") && snapshot.first.optString("state") in
                                listOf("cpu_hold", "suspending") -> snapshot.first.optString("detail")
                            !control.has("memory_hold_reason") -> ""
                            snapshot.first.optString("state") == "memory_hold" && snapshot.first.optBoolean("retry_as_new") ->
                                "Training stopped before a readable checkpoint. Use Retry as new task."
                            snapshot.first.optString("state") == "memory_hold" ->
                                "Training stopped to free memory. When ready, tap Start / Resume."
                            snapshot.first.optString("state") == "suspending" ->
                                "Giving other apps memory. Waiting for the trainer to save and exit."
                            else -> ""
                        }
                        retryAsNew = snapshot.first.optBoolean("retry_as_new")
                        loadedTask = id
                    } catch (e: Exception) {
                        if (e is kotlinx.coroutines.CancellationException) throw e
                        status = "Cannot read task: ${e.message}"
                        loadedTask = null
                    }
                    delay(1000)
                }
            }
            MaterialTheme {
                if (confirmFocus) AlertDialog(onDismissRequest = { confirmFocus = false },
                    title = { Text("Pause all training for one hour?") },
                    text = { Text("Release the active trainer and block all tasks. Work after the last checkpoint may be replayed. Training does not restart by itself when focus ends.") },
                    confirmButton = { TextButton(onClick = {
                        focus.start()
                        focusMs = focus.withGate { it.remainingMs }
                        confirmFocus = false
                    }) { Text("Start focus") } },
                    dismissButton = { TextButton(onClick = { confirmFocus = false }) { Text("Cancel") } })
                confirmRelease?.let { id ->
                    AlertDialog(onDismissRequest = { confirmRelease = null },
                        title = { Text("Release training resources now?") },
                        text = { Text("The unfinished update is discarded. Resume uses the last committed checkpoint. Before the first checkpoint you may have to start again.") },
                        confirmButton = { TextButton(onClick = {
                            pauseTask(id, immediately = true)
                            confirmRelease = null
                        }) { Text("Release now") } },
                        dismissButton = { TextButton(onClick = { confirmRelease = null }) { Text("Cancel") } })
                }
                Surface(Modifier.fillMaxSize()) {
                    Column(Modifier.windowInsetsPadding(WindowInsets.safeDrawing).padding(24.dp).verticalScroll(rememberScrollState()), verticalArrangement = Arrangement.spacedBy(12.dp)) {
                        Text("MobileFineTuner Continuity", style = MaterialTheme.typography.headlineSmall)
                        OutlinedButton(onClick = {
                            if (focusMs > 0) {
                                focus.end()
                                focusMs = 0L
                                message = "Focus ended. Tap Start / Resume when ready."
                            } else confirmFocus = true
                        }) { Text(if (focusMs > 0L) "End focus (${(focusMs + 59999) / 60000} min left)" else "Focus for one hour (release all training)") }
                        Text("Pause saves after a complete optimizer update. Resume restores the original configuration, optimizer, data cursor and metrics. A forced kill recovers only the latest committed update.")
                        Button(enabled = !importing, onClick = { importer.launch(null) }) { Text(if (importing) "Importing..." else "Import experiment folder") }
                        Button(enabled = !importing, onClick = { scope.launch {
                            try { selected = withContext(Dispatchers.IO) { store.createDemo() }; ids = store.list() }
                            catch (e: Exception) { message = e.message ?: "Demo failed" }
                        } }) { Text("Create tiny Qwen test task (not a benchmark)") }
                        Box {
                            TextButton(onClick = { menu = true }) { Text("Task: ${selected ?: "none"}") }
                            DropdownMenu(expanded = menu, onDismissRequest = { menu = false }) {
                                ids.forEach { id -> DropdownMenuItem(text = { Text(id) }, onClick = { selected = id; menu = false }) }
                            }
                        }
                        if (pressureMessage.isNotEmpty()) Text(pressureMessage, style = MaterialTheme.typography.titleMedium)
                        if (progressLine.isNotEmpty()) Text(progressLine)
                        if (showBatch && loadedTask == selected) {
                            Text("Resume physical batch", style = MaterialTheme.typography.titleMedium)
                            Text("Last checkpoint: up to $committedBatch examples at once. A smaller cap reduces memory use. Effective batch stays $effectiveBatch. Memory checks still apply.")
                            if (originalBatch > 32) {
                                OutlinedTextField(value = batchChoice?.toString().orEmpty(), onValueChange = { input ->
                                    if (input.isEmpty()) batchChoice = null
                                    else input.toIntOrNull()?.takeIf { it in 1..originalBatch }?.let { batchChoice = it }
                                }, label = { Text("Cap 1-$originalBatch (blank: keep $previousBatch)") }, singleLine = true)
                            } else Box {
                                OutlinedButton(onClick = { batchMenu = true }) {
                                    Text(batchChoice?.let { "Physical batch: $it" } ?: "Keep previous batch ($previousBatch)")
                                }
                                DropdownMenu(expanded = batchMenu, onDismissRequest = { batchMenu = false }) {
                                    DropdownMenuItem(text = { Text("Keep previous setting") }, onClick = {
                                        batchChoice = null; batchMenu = false
                                    })
                                    (1..originalBatch).forEach { size ->
                                        DropdownMenuItem(text = { Text("Use physical batch $size") }, onClick = {
                                            batchChoice = size; batchMenu = false
                                        })
                                    }
                                }
                            }
                        }
                        Text(status)
                        Row { Checkbox(automatic, { automatic = it }); Text("Continue with scheduled background slices") }
                        Row { Checkbox(charging, { charging = it }); Text("Train only while charging") }
                        val canEditCpu = selected != null && loadedTask == selected && !starting && !cpuSaving
                        Row(Modifier.fillMaxWidth().toggleable(value = cpuGuard, enabled = canEditCpu,
                            role = Role.Switch, onValueChange = { enabled ->
                                val id = selected ?: return@toggleable
                                val previous = cpuGuard
                                cpuGuard = enabled
                                cpuSaving = true
                                cpuEditVersion++
                                val write = admissions.async { store.setCpuContentionGuard(id, enabled) }
                                scope.launch {
                                    try { write.await() }
                                    catch (e: kotlinx.coroutines.CancellationException) { throw e }
                                    catch (e: Exception) {
                                        cpuGuard = previous
                                        if (selected == id) message = e.message ?: "Could not save CPU setting"
                                    } finally {
                                        cpuEditVersion++ // Discard any poll started before the write finished.
                                        cpuSaving = false
                                    }
                                }
                            }).heightIn(min = 48.dp), verticalAlignment = Alignment.CenterVertically) {
                            Text("Pause for CPU contention", Modifier.weight(1f))
                            Switch(checked = cpuGuard, onCheckedChange = null, enabled = canEditCpu)
                        }
                        Text("Applies to this task's next training slice. If the trainer keeps waiting for CPU, it saves and stops. Resume manually.")
                        OutlinedTextField(value = sliceText, onValueChange = { sliceText = it },
                            label = { Text("Slice seconds (1-120; runtime only)") }, singleLine = true)
                        OutlinedTextField(value = graceText, onValueChange = { graceText = it },
                            label = { Text("Pause grace seconds (30-420; runtime only)") }, singleLine = true)
                        Button(enabled = focusMs == 0L && selected != null && loadedTask == selected && !retryAsNew && !importing && !starting && !cpuSaving, onClick = {
                            val id = selected ?: return@Button
                            val slice = sliceText.toIntOrNull()
                            if (slice == null || slice !in 1..120) { message = "Slice must be 1..120 seconds"; return@Button }
                            val grace = graceText.toIntOrNull()
                            if (grace == null || grace !in 30..420) { message = "Pause grace must be 30..420 seconds"; return@Button }
                            starting = true
                            val issued = android.os.SystemClock.elapsedRealtimeNanos()
                            val requestedAutomatic = automatic
                            val requestedCharging = charging
                            val requestedBatch = batchChoice.takeIf { showBatch }
                            val app = applicationContext
                            val taskStore = TaskStore(app)
                            if (Build.VERSION.SDK_INT >= 33) permissions.launch(Manifest.permission.POST_NOTIFICATIONS)
                            val admission = admissions.async {
                                var generation: Long? = null
                                try {
                                    generation = taskStore.admitStart(id, requestedAutomatic, requestedCharging, issued, slice, grace, requestedBatch)
                                        ?: return@async null
                                    if (taskStore.anotherTaskExecuting(id)) {
                                        taskStore.rejectStart(id, generation)
                                        return@async "Start rejected: another task is active. Pause it and retry; this request was not queued."
                                    }
                                    if (requestedAutomatic) Scheduling.schedule(app, id, generation, replace = true)
                                    else Scheduling.cancel(app, id, generation)
                                    taskStore.dispatchStartIfCurrent(id, generation) {
                                        ContextCompat.startForegroundService(app, Intent(app, TrainingService::class.java)
                                            .putExtra("task", id).putExtra("resume_generation", generation))
                                    }
                                    null
                                } catch (e: Exception) {
                                    generation?.let { if (!taskStore.fail(id, e.message ?: "Start failed", it, onlyIfEnabled = true)) return@async null }
                                    e.message ?: "Start failed"
                                }
                            }
                            scope.launch {
                                try { admission.await()?.let { message = it } }
                                finally { starting = false }
                            }
                        }) { Text("Start / Resume") }
                        if (retryAsNew) {
                            Text("Exact resume is unavailable. Retry creates a new task with the same inputs and configuration. Original checkpoints and diagnostics are kept.")
                            OutlinedButton(enabled = !importing && !starting && loadedTask == selected, onClick = {
                                val id = selected ?: return@OutlinedButton
                                importing = true
                                scope.launch {
                                    try {
                                        selected = withContext(Dispatchers.IO) { store.retryAsNew(id) }
                                        ids = store.list()
                                        message = "New task ready. Original task $id and its diagnostics are retained. Press Start / Resume to reinitialize."
                                    } catch (e: Exception) { message = e.message ?: "Retry creation failed" }
                                    finally { importing = false }
                                }
                            }) { Text("Retry as new task (reinitialize)") }
                        }
                        OutlinedButton(enabled = selected != null, onClick = {
                            selected?.let { pauseTask(it) }
                        }) { Text("Pause and disable background continuation") }
                        OutlinedButton(enabled = selected != null && loadedTask == selected, onClick = {
                            confirmRelease = selected
                        }) { Text("Release resources now (replay unfinished update)") }
                        OutlinedButton(enabled = selected != null, onClick = { exporter.launch("mft-$selected.zip") }) { Text("Export paused task records") }
                        Text(message)
                        Text("Each slice is bounded. Low battery, heat, memory or storage pressure requests a safe pause. Android may delay scheduled work; force-stop requires reopening the app. Do not update the APK or move model files during an unfinished task.")
                        TextButton(onClick = { startActivity(Intent(this@TrainingActivity, MainActivity::class.java)) }) { Text("Open existing log visualizer") }
                    }
                }
            }
        }
    }
}
