package com.mobilefinetuner.sdk;

import java.io.File;
import java.util.Arrays;

/**
 * Java-facing Android SDK for MobileFineTuner.
 *
 * <p>The SDK owns one native MF session at a time: model, optional LoRA
 * adapters, and an optimizer-backed trainer. Model weights are supplied by the
 * application as files on device storage; they are not bundled in the AAR.</p>
 */
public final class MobileFineTuner implements AutoCloseable {
    static {
        System.loadLibrary("mobilefinetuner_jni");
    }

    private long nativeHandle;

    private MobileFineTuner(long nativeHandle) {
        if (nativeHandle == 0L) {
            throw new IllegalStateException("Native MobileFineTuner session was not created");
        }
        this.nativeHandle = nativeHandle;
    }

    public static String buildInfo() {
        return nativeBuildInfo();
    }

    public static SelfTestResult selfTest(File workingDir) {
        if (workingDir == null) {
            throw new IllegalArgumentException("workingDir must not be null");
        }
        double[] result = nativeSelfTest(workingDir.getAbsolutePath());
        return new SelfTestResult((float) result[0], (int) result[1], result[2]);
    }

    public static MobileFineTuner open(String modelDir) {
        return open(modelDir, true);
    }

    public static MobileFineTuner open(String modelDir, boolean loadWeights) {
        if (modelDir == null || modelDir.isEmpty()) {
            throw new IllegalArgumentException("modelDir must not be empty");
        }
        return new MobileFineTuner(nativeCreate(modelDir, loadWeights));
    }

    public void initLora(LoraConfig config) {
        if (config == null) {
            throw new IllegalArgumentException("config must not be null");
        }
        nativeInitLora(
                requireOpen(),
                config.rank,
                config.alpha,
                config.dropout,
                config.seed,
                config.targetModules
        );
    }

    public void createTrainer(TrainerConfig config) {
        if (config == null) {
            throw new IllegalArgumentException("config must not be null");
        }
        nativeCreateTrainer(
                requireOpen(),
                config.learningRate,
                config.weightDecay,
                config.maxGradNorm,
                config.ignoreIndex
        );
    }

    public TrainStepResult trainStep(
            int[] inputIds,
            float[] attentionMask,
            int[] labels,
            int batchSize,
            int sequenceLength
    ) {
        if (batchSize <= 0 || sequenceLength <= 1) {
            throw new IllegalArgumentException("batchSize must be positive and sequenceLength must be > 1");
        }
        int expected = batchSize * sequenceLength;
        requireLength(inputIds, expected, "inputIds");
        requireLength(attentionMask, expected, "attentionMask");
        requireLength(labels, expected, "labels");

        double[] result = nativeTrainStep(
                requireOpen(),
                inputIds,
                attentionMask,
                labels,
                batchSize,
                sequenceLength
        );
        return new TrainStepResult((float) result[0], (int) result[1]);
    }

    public int trainableTensorCount() {
        return nativeTrainableTensorCount(requireOpen());
    }

    @Override
    public void close() {
        long handle = nativeHandle;
        nativeHandle = 0L;
        if (handle != 0L) {
            nativeClose(handle);
        }
    }

    private long requireOpen() {
        if (nativeHandle == 0L) {
            throw new IllegalStateException("MobileFineTuner session is closed");
        }
        return nativeHandle;
    }

    private static void requireLength(Object array, int expected, String name) {
        if (array == null) {
            throw new IllegalArgumentException(name + " must not be null");
        }
        int length;
        if (array instanceof int[]) {
            length = ((int[]) array).length;
        } else if (array instanceof float[]) {
            length = ((float[]) array).length;
        } else {
            throw new IllegalArgumentException(name + " has unsupported array type");
        }
        if (length != expected) {
            throw new IllegalArgumentException(
                    name + " length must be " + expected + ", got " + length
            );
        }
    }

    private static native String nativeBuildInfo();
    private static native double[] nativeSelfTest(String workingDir);
    private static native long nativeCreate(String modelDir, boolean loadWeights);
    private static native void nativeInitLora(
            long handle,
            int rank,
            float alpha,
            float dropout,
            long seed,
            String[] targetModules
    );
    private static native void nativeCreateTrainer(
            long handle,
            float learningRate,
            float weightDecay,
            float maxGradNorm,
            int ignoreIndex
    );
    private static native double[] nativeTrainStep(
            long handle,
            int[] inputIds,
            float[] attentionMask,
            int[] labels,
            int batchSize,
            int sequenceLength
    );
    private static native int nativeTrainableTensorCount(long handle);
    private static native void nativeClose(long handle);

    public static final class LoraConfig {
        public final int rank;
        public final float alpha;
        public final float dropout;
        public final long seed;
        public final String[] targetModules;

        public LoraConfig(int rank, float alpha, float dropout, long seed, String[] targetModules) {
            if (rank <= 0) {
                throw new IllegalArgumentException("rank must be positive");
            }
            if (alpha <= 0.0f) {
                throw new IllegalArgumentException("alpha must be positive");
            }
            if (dropout < 0.0f || dropout >= 1.0f) {
                throw new IllegalArgumentException("dropout must be in [0, 1)");
            }
            this.rank = rank;
            this.alpha = alpha;
            this.dropout = dropout;
            this.seed = seed;
            this.targetModules = targetModules == null
                    ? new String[0]
                    : Arrays.copyOf(targetModules, targetModules.length);
        }

        public static LoraConfig attentionQkvo() {
            return new LoraConfig(
                    8,
                    16.0f,
                    0.05f,
                    42L,
                    new String[]{"q_proj", "k_proj", "v_proj", "o_proj"}
            );
        }
    }

    public static final class TrainerConfig {
        public final float learningRate;
        public final float weightDecay;
        public final float maxGradNorm;
        public final int ignoreIndex;

        public TrainerConfig(float learningRate, float weightDecay, float maxGradNorm, int ignoreIndex) {
            if (learningRate <= 0.0f) {
                throw new IllegalArgumentException("learningRate must be positive");
            }
            if (weightDecay < 0.0f) {
                throw new IllegalArgumentException("weightDecay must be non-negative");
            }
            if (maxGradNorm <= 0.0f) {
                throw new IllegalArgumentException("maxGradNorm must be positive");
            }
            this.learningRate = learningRate;
            this.weightDecay = weightDecay;
            this.maxGradNorm = maxGradNorm;
            this.ignoreIndex = ignoreIndex;
        }

        public static TrainerConfig defaults() {
            return new TrainerConfig(2e-4f, 0.0f, 1.0f, -100);
        }
    }

    public static final class TrainStepResult {
        public final float loss;
        public final int trainableTensorCount;

        private TrainStepResult(float loss, int trainableTensorCount) {
            this.loss = loss;
            this.trainableTensorCount = trainableTensorCount;
        }
    }

    public static final class SelfTestResult {
        public final float loss;
        public final int trainableTensorCount;
        public final double elapsedMillis;

        private SelfTestResult(float loss, int trainableTensorCount, double elapsedMillis) {
            this.loss = loss;
            this.trainableTensorCount = trainableTensorCount;
            this.elapsedMillis = elapsedMillis;
        }
    }
}
