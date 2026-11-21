package com.mobilefinetuner.visualizer.continuity

import java.io.File
import java.io.RandomAccessFile

internal data class WorkPolicy(val generation: Long, val automatic: Boolean, val charging: Boolean)

internal object GenerationWorkGate {
    private val monitors = Array(16) { Any() }

    // Only WM writers take this lock. read() must release task control before returning:
    // cancellation can wait for a worker which is itself reading/writing task control.
    fun apply(task: File, generation: Long, automatic: Boolean, read: () -> WorkPolicy,
              mutate: (WorkPolicy) -> Unit): Boolean = synchronized(monitors[task.path.hashCode() and 15]) {
        RandomAccessFile(File(task, "schedule.lock"), "rw").use { file ->
            file.channel.lock().use {
                val policy = read()
                if (policy.generation != generation || policy.automatic != automatic) false
                else { mutate(policy); true }
            }
        }
    }
}
