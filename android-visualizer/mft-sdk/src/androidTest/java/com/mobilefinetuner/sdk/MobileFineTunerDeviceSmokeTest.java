package com.mobilefinetuner.sdk;

import static org.junit.Assert.assertTrue;

import android.content.Context;

import androidx.test.core.app.ApplicationProvider;
import androidx.test.ext.junit.runners.AndroidJUnit4;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;

import org.junit.Test;
import org.junit.runner.RunWith;

@RunWith(AndroidJUnit4.class)
public final class MobileFineTunerDeviceSmokeTest {
    @Test
    public void nativeLibraryLoads() {
        String buildInfo = MobileFineTuner.buildInfo();
        assertTrue(buildInfo.contains("MobileFineTuner Android SDK JNI"));
    }

    @Test
    public void nativeSelfTestRunsOneTrainingStep() {
        Context context = ApplicationProvider.getApplicationContext();
        MobileFineTuner.SelfTestResult result = MobileFineTuner.selfTest(context.getFilesDir());
        assertTrue(Float.isFinite(result.loss));
        assertTrue(result.trainableTensorCount > 0);
        assertTrue(result.elapsedMillis >= 0.0);
    }
    @Test
    public void concurrentSelfTestsUseSerializedNativeRuntime() throws Exception {
        Context context = ApplicationProvider.getApplicationContext();
        ExecutorService workers = Executors.newFixedThreadPool(2);
        try {
            Future<MobileFineTuner.SelfTestResult> first = workers.submit(
                    () -> MobileFineTuner.selfTest(context.getFilesDir()));
            Future<MobileFineTuner.SelfTestResult> second = workers.submit(
                    () -> MobileFineTuner.selfTest(context.getFilesDir()));
            assertTrue(Float.isFinite(first.get(30, TimeUnit.SECONDS).loss));
            assertTrue(Float.isFinite(second.get(30, TimeUnit.SECONDS).loss));
        } finally {
            workers.shutdownNow();
        }
    }

}
