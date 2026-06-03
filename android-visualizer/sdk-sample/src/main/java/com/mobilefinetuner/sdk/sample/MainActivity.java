package com.mobilefinetuner.sdk.sample;

import android.app.Activity;
import android.os.Bundle;
import android.util.Log;
import android.widget.TextView;

import com.mobilefinetuner.sdk.MobileFineTuner;

import java.util.Locale;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public final class MainActivity extends Activity {
    private static final String TAG = "MFTSdkSample";

    private final ExecutorService executor = Executors.newSingleThreadExecutor();

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        TextView view = new TextView(this);
        view.setTextSize(16.0f);
        view.setPadding(32, 32, 32, 32);
        view.setText("MobileFineTuner SDK\n" + MobileFineTuner.buildInfo() + "\n\nRunning native self-test...");
        setContentView(view);

        executor.execute(() -> {
            String message = runSelfTestMessage();
            runOnUiThread(() -> view.setText(message));
        });
    }

    private String runSelfTestMessage() {
        try {
            MobileFineTuner.SelfTestResult result = MobileFineTuner.selfTest(getFilesDir());
            Log.i(TAG, "Self-test passed: loss=" + result.loss
                    + ", trainable_tensors=" + result.trainableTensorCount
                    + ", elapsed_ms=" + result.elapsedMillis);
            return String.format(
                    Locale.US,
                    "MobileFineTuner SDK\n%s\n\nSelf-test passed\nloss=%.6f\ntrainable_tensors=%d\nelapsed_ms=%.2f",
                    MobileFineTuner.buildInfo(),
                    result.loss,
                    result.trainableTensorCount,
                    result.elapsedMillis
            );
        } catch (Throwable t) {
            Log.e(TAG, "Self-test failed", t);
            return "MobileFineTuner SDK\nSelf-test failed\n" + t.getMessage();
        }
    }

    @Override
    protected void onDestroy() {
        executor.shutdownNow();
        super.onDestroy();
    }
}
