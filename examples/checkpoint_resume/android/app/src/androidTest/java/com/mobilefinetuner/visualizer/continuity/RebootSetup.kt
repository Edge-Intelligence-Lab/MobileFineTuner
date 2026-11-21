package com.mobilefinetuner.visualizer.continuity

import androidx.test.core.app.ApplicationProvider
import android.content.Context
import org.json.JSONObject
import org.junit.Test
import java.io.File

/** Executed explicitly by tests/android_reboot.py, not named *Test in the regular suite. */
class RebootSetup {
    @Test fun prepare() {
        requireExplicitFixture(this)
        val store = TaskStore(ApplicationProvider.getApplicationContext<Context>())
        val id = store.createDemo()
        TaskStore.write(File(store.task(id), "spec.json"), store.spec(id).put("steps", 300).put("slice_seconds", 1))
        val generation = store.allow(id, true, false)
        TaskStore.write(File(store.context.filesDir, "reboot_task.json"), JSONObject().put("task", id))
        Scheduling.schedule(store.context, id, generation)
    }
}
