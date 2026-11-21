package com.mobilefinetuner.visualizer.continuity

import android.content.Context
import android.os.Build
import android.os.SystemClock
import android.util.Log
import androidx.test.core.app.ApplicationProvider
import androidx.test.platform.app.InstrumentationRegistry
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Assume.assumeTrue
import org.junit.Test
import java.io.File
import java.util.concurrent.FutureTask
import java.util.concurrent.TimeUnit

class CpuContentionPilotTest {
    @Test fun pinnedTrainerAloneThenWithTwoBoundedCompetitors() {
        assumeTrue(InstrumentationRegistry.getArguments().getString("cpuPilot") == "true")
        assumeTrue(Build.HARDWARE in listOf("ranchu", "goldfish") && "arm64-v8a" in Build.SUPPORTED_ABIS)
        val context = ApplicationProvider.getApplicationContext<Context>()
        val store = TaskStore(context)
        val runner = TrainingRunner(context)
        val ram = MemoryGuard(128)
        fun healthy() {
            val r = Resources.read(context)
            check(ram.beforeStart(r) == null && r.nonMemoryBlock(false) == null) {
                "Not a healthy-RAM CPU trial: $r"
            }
        }
        healthy()
        val id = store.createDemo()
        val dir = store.task(id)
        val run = File(dir, "run")
        TaskStore.write(File(dir, "spec.json"), store.spec(id).put("cpu_contention_guard", true)
            .put("threads", 1).put("min_available_mb", 128).put("steps", 20000)
            .put("seq_len", 64).put("slice_seconds", 120))
        val counters = ChildScheduler(run, 1)
        val gen = store.allow(id, true, false)
        val task = FutureTask { runner.run(id, true, expectedGeneration = gen) }
        val burners = mutableListOf<Process>()
        Thread(task, "cpu-pilot-trainer").start()
        try {
            val start = SystemClock.elapsedRealtime()
            while ((store.checkpoint(id)?.optInt("committed_updates") ?: 0) < 1) {
                check(!task.isDone && SystemClock.elapsedRealtime() - start < 20000) { store.status(id).toString() }
                SystemClock.sleep(50)
            }
            val session = File(run, "sessions").listFiles()!!.single()
            val pid = File(session, "events.jsonl").bufferedReader().use { JSONObject(it.readLine()).getInt("pid") }
            checkNotNull(counters.read()) { "Cannot validate and sample the live trainer" }
            val cpu = File("/proc/$pid/status").readLines().single { it.startsWith("Cpus_allowed_list:") }
                .substringAfter(':').trim().substringBefore(',').substringBefore('-').toInt()
            check(cpu in 0..62)
            val mask = java.lang.Long.toHexString(1L shl cpu)
            val pin = ProcessBuilder("/system/bin/taskset", "-p", mask, pid.toString()).start()
            check(pin.waitFor(3, TimeUnit.SECONDS) && pin.exitValue() == 0) { "Cannot pin trainer to CPU $cpu" }

            fun measure(label: String, a: SchedulerTimes, b: SchedulerTimes): Double {
                val runNs = b.runNs - a.runNs
                val waitNs = b.waitNs - a.waitNs
                check(runNs > 0 && waitNs >= 0)
                val share = waitNs.toDouble() / (runNs + waitNs)
                Log.i("MftCpuPilot", "$label cpu=$cpu run_ns=$runNs wait_ns=$waitNs wait_share=$share")
                return share
            }
            val aloneStart = checkNotNull(counters.read())
            val updatesBefore = store.checkpoint(id)!!.getInt("committed_updates")
            repeat(24) {
                SystemClock.sleep(250)
                healthy()
                check(!task.isDone && store.control(id).getBoolean("enabled")) { "False hold in solo phase" }
            }
            check(store.checkpoint(id)!!.getInt("committed_updates") > updatesBefore)
            val loadStart = checkNotNull(counters.read())
            val solo = measure("solo", aloneStart, loadStart)
            repeat(2) {
                // Android mksh built-ins only. exec through taskset; self-expire even if instrumentation dies.
                burners += ProcessBuilder("/system/bin/taskset", mask, "/system/bin/sh", "-c",
                    "end=\$((SECONDS + 18)); while (( SECONDS < end )); do :; done").start()
            }
            val loadAt = SystemClock.elapsedRealtime()
            var last = loadStart
            while (!task.isDone) {
                healthy()
                counters.read()?.let { last = it }
                check(burners.all { it.isAlive } && SystemClock.elapsedRealtime() - loadAt < 16000) {
                    "No CPU hold with live competitors: ${store.status(id)}"
                }
                SystemClock.sleep(100)
            }
            check(measure("competing", loadStart, last) > solo)
            assertEquals(TrainingRunner.Outcome.SUSPENDED, task.get())
            val status = store.status(id)
            assertEquals("cpu_hold", status.getString("state"))
            assertEquals("suspended", status.getString("committed_status"))
            assertFalse(status.getBoolean("execution_active"))
            assertFalse(store.canRetryInBackground(id, gen))
            Log.i("MftCpuPilot", status.getString("detail"))
            burners.forEach { it.destroyForcibly().waitFor() }
            val stopped = status.getInt("committed_updates")
            assertEquals(TrainingRunner.Outcome.DISABLED, runner.run(id, true))
            store.allow(id, false, false)
            assertFalse(store.control(id).has("cpu_hold_reason"))
            assertEquals(TrainingRunner.Outcome.SUSPENDED, runner.run(id, false, stopAfter = stopped + 1))
            assertEquals(stopped + 1, store.checkpoint(id)!!.getInt("committed_updates"))
        } finally {
            burners.forEach { if (it.isAlive) it.destroyForcibly().waitFor() }
            store.pause(id, immediately = true)
            task.get(155, TimeUnit.SECONDS)
            dir.deleteRecursively()
        }
    }
}
