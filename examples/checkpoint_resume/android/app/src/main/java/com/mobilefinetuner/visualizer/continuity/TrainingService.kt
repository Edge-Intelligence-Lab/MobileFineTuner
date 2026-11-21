package com.mobilefinetuner.visualizer.continuity

import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.BroadcastReceiver
import android.os.IBinder
import androidx.core.app.NotificationCompat
import androidx.work.Constraints
import androidx.work.ExistingWorkPolicy
import androidx.work.OneTimeWorkRequestBuilder
import androidx.work.WorkManager
import androidx.work.Worker
import androidx.work.WorkerParameters
import androidx.work.workDataOf
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean

class TrainingBootReceiver : BroadcastReceiver() {
    override fun onReceive(context: Context, intent: Intent) {
        if (intent.action != Intent.ACTION_BOOT_COMPLETED) return
        val pending = goAsync()
        Thread {
            try {
                val store = TaskStore(context)
                for (id in store.list()) {
                    var generation: Long? = null
                    try {
                        store.respectSystemStop(id)
                        generation = store.control(id).optLong("resume_generation")
                        if (store.status(id).optString("committed_status") != "completed")
                            Scheduling.schedule(context, id, generation)
                    } catch (e: Exception) {
                        generation?.let { store.note(id, "waiting", "Boot scheduling failed: ${e.message}", it) }
                    }
                }
            } finally { pending.finish() }
        }.start()
    }
}

object Scheduling {
    const val GENERATION = "resume_generation"
    internal fun tag(generation: Long) = "mft-generation-$generation"
    private fun mutate(context: Context, id: String, generation: Long, automatic: Boolean,
                       action: (WorkManager, WorkPolicy) -> Unit): Boolean {
        val store = TaskStore(context)
        return GenerationWorkGate.apply(store.task(id), generation, automatic, {
            val ctl = store.control(id)
            WorkPolicy(ctl.optLong(GENERATION), ctl.optBoolean("enabled") && ctl.optBoolean("automatic"),
                ctl.optBoolean("charging", true))
        }) { action(WorkManager.getInstance(context), it) }
    }
    fun schedule(context: Context, id: String, generation: Long, replace: Boolean = false): Boolean = mutate(
        context, id, generation, automatic = true
    ) { wm, policy ->
        val name = "mft-$id"
        // KEEP only compatible work. This also replaces unversioned pre-upgrade requests.
        val obsolete = wm.getWorkInfosForUniqueWork(name).get().any { !it.state.isFinished && tag(generation) !in it.tags }
        val request = OneTimeWorkRequestBuilder<TrainingWorker>()
            .addTag(tag(generation))
            .setInputData(workDataOf("task" to id, GENERATION to generation))
            // Persist the handoff first, but give the visible foreground slice first use of the lease.
            .setInitialDelay(5, TimeUnit.SECONDS)
            .setConstraints(Constraints.Builder().setRequiresCharging(policy.charging)
                .setRequiresBatteryNotLow(true).setRequiresStorageNotLow(true).build())
            .setBackoffCriteria(androidx.work.BackoffPolicy.LINEAR, 10, TimeUnit.SECONDS).build()
        wm.enqueueUniqueWork(name, if (replace || obsolete) ExistingWorkPolicy.REPLACE else ExistingWorkPolicy.KEEP,
            request).result.get()
    }

    fun cancel(context: Context, id: String, generation: Long): Boolean = mutate(context, id, generation, false) { wm, _ ->
        wm.cancelUniqueWork("mft-$id").result.get()
    }
}

class TrainingWorker(context: Context, params: WorkerParameters) : Worker(context, params) {
    override fun doWork(): Result {
        val id = inputData.getString("task") ?: return Result.failure()
        val store = TaskStore(applicationContext)
        val generation = inputData.getLong(Scheduling.GENERATION, -1L)
        if (generation < 1 || !store.canRetryInBackground(id, generation)) return Result.success()
        return try {
            when (TrainingRunner(applicationContext).run(id, true, { isStopped }, expectedGeneration = generation)) {
                TrainingRunner.Outcome.COMPLETED, TrainingRunner.Outcome.DISABLED -> Result.success()
                TrainingRunner.Outcome.FAILED -> Result.failure()
                TrainingRunner.Outcome.WAITING, TrainingRunner.Outcome.SUSPENDED, TrainingRunner.Outcome.BUSY ->
                    if (store.canRetryInBackground(id, generation)) Result.retry() else Result.success()
            }
        } catch (e: Exception) {
            if (isStopped) return Result.success() // Cancellation must not revoke a newer user Resume.
            store.fail(id, e.message ?: e.javaClass.simpleName, generation)
            Result.failure()
        }
    }
}

class TrainingService : Service() {
    companion object {
        // One old slice's maximum execution budget plus its existing cleanup allowance.
        const val HANDOFF_WAIT_MS = 550000L
        fun handoffExpired(startedMs: Long, nowMs: Long): Boolean = nowMs - startedMs >= HANDOFF_WAIT_MS
        fun openTaskIntent(context: Context, id: String): Intent = Intent(context, TrainingActivity::class.java)
            .setAction("com.mobilefinetuner.OPEN_TASK.$id").putExtra("task", id)
            .addFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP or Intent.FLAG_ACTIVITY_SINGLE_TOP)
    }
    private val active = AtomicBoolean(false)
    private val stopping = AtomicBoolean(false)
    private var taskId: String? = null
    // Accessed only on the main thread; this is one same-task slot, not a task queue.
    private var requestedGeneration = -1L
    private var lastStartId = 0
    override fun onBind(intent: Intent?): IBinder? = null
    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        lastStartId = startId
        val id = intent?.getStringExtra("task") ?: return START_NOT_STICKY
        if (intent.action == "pause") {
            TaskStore(this).pauseIfCurrent(id, intent.getLongExtra(Scheduling.GENERATION, -1L))
            if (!active.get()) stopSelfResult(startId)
            return START_NOT_STICKY
        }
        val store = TaskStore(this)
        val generation = if (intent.hasExtra("resume_generation")) intent.getLongExtra("resume_generation", 0)
            else store.control(id).optLong("resume_generation")
        if (!active.compareAndSet(false, true)) {
            if (id != taskId) store.rejectStart(id, generation)
            else if (generation > requestedGeneration) {
                requestedGeneration = generation
                showForeground(id, generation)
                store.note(id, "handoff", "New Resume retained; waiting for the previous same-task execution to exit.", generation)
            }
            return START_NOT_STICKY
        }
        taskId = id
        requestedGeneration = generation
        showForeground(id, generation)
        startGeneration(id, generation)
        return START_NOT_STICKY
    }
    private fun showForeground(id: String, generation: Long) {
        val manager = getSystemService(NotificationManager::class.java)
        manager.createNotificationChannel(NotificationChannel("training", "Model training", NotificationManager.IMPORTANCE_LOW))
        val open = PendingIntent.getActivity(this, 0, openTaskIntent(this, id),
            PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT)
        val pause = PendingIntent.getService(this, 1, Intent(this, TrainingService::class.java)
            .setAction("pause").setData(android.net.Uri.parse("mft://task/$id/$generation"))
            .putExtra("task", id).putExtra(Scheduling.GENERATION, generation),
            PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT)
        startForeground(41, NotificationCompat.Builder(this, "training")
            .setSmallIcon(android.R.drawable.stat_notify_sync).setContentTitle("MobileFineTuner training")
            .setContentText("Finite training slice; Pause saves at the next update")
            .setContentIntent(open).setOngoing(true).addAction(0, "Pause", pause).build())
    }
    private fun startGeneration(id: String, generation: Long) {
        Thread {
            val store = TaskStore(this)
            try {
                runWithHandoff(store, id, generation)
            } catch (e: Exception) {
                store.fail(id, e.message ?: e.javaClass.simpleName, generation)
            } finally {
                android.os.Handler(mainLooper).post {
                    val control = store.control(id)
                    val next = requestedGeneration
                    // Recheck on the same thread that accepts intents so a late Resume is not lost at teardown.
                    if (!stopping.get() && next > generation && control.optBoolean("enabled") &&
                        control.optLong("resume_generation") == next) {
                        startGeneration(id, next)
                        return@post
                    }
                    stopForeground(STOP_FOREGROUND_REMOVE)
                    active.set(false)
                    stopSelfResult(lastStartId)
                }
            }
        }.start()
    }
    private fun runWithHandoff(store: TaskStore, id: String, generation: Long) {
        val started = android.os.SystemClock.elapsedRealtime()
        var sameTaskObserved = store.isExecuting(id)
        while (!stopping.get()) {
            val control = store.control(id)
            if (!control.optBoolean("enabled") || control.optLong("resume_generation") != generation) return
            val outcome = TrainingRunner(this).run(id, false, { stopping.get() },
                expectedGeneration = generation, rejectBusy = false)
            if (outcome != TrainingRunner.Outcome.BUSY) return
            val sameTaskActive = store.isExecuting(id)
            sameTaskObserved = sameTaskObserved || sameTaskActive
            // Allow the old owner's active/global lock release gap, but never queue behind another task.
            if (!sameTaskActive && (!sameTaskObserved || store.list().any { it != id && store.isExecuting(it) })) {
                store.rejectStart(id, generation)
                return
            }
            if (handoffExpired(started, android.os.SystemClock.elapsedRealtime())) {
                store.fail(id, "Same-task Resume handoff timed out; previous execution did not release its lease. Retry explicitly.", generation)
                return
            }
            store.note(id, "handoff", "Resume retained; waiting for the previous same-task lease (at most 550 seconds).", generation)
            Thread.sleep(200)
        }
    }
    override fun onDestroy() {
        stopping.set(true)
        // The owning Runner observes stopping; a completed service must not signal a newer slice.
        super.onDestroy()
    }
}
