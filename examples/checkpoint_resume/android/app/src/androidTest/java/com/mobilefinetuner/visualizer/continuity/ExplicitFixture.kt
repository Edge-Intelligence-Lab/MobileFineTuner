package com.mobilefinetuner.visualizer.continuity

import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assume.assumeTrue

// These routines need staged assets or an external lifecycle driver. Never
// schedule a boot task as a side effect of ordinary instrumentation discovery.
internal fun requireExplicitFixture(instance: Any) {
    val selection = InstrumentationRegistry.getArguments().getString("class").orEmpty()
        .split(',').map { it.substringBefore('#') }
    assumeTrue("Use the matching host fixture driver", instance.javaClass.name in selection)
}
