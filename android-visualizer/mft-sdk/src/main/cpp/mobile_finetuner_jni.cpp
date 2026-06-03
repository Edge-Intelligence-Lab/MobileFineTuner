#include <jni.h>

#include "mobile_finetuner/mobile_finetuner.h"

#include <chrono>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct NativeSession {
    std::unique_ptr<ops::AutoModelForCausalLM> model;
    std::unique_ptr<ops::AutoTrainer> trainer;
};

void throw_java(JNIEnv* env, const char* class_name, const std::string& message) {
    jclass clazz = env->FindClass(class_name);
    if (clazz != nullptr) {
        env->ThrowNew(clazz, message.c_str());
    }
}

void throw_illegal_state(JNIEnv* env, const std::string& message) {
    throw_java(env, "java/lang/IllegalStateException", message);
}

void throw_illegal_argument(JNIEnv* env, const std::string& message) {
    throw_java(env, "java/lang/IllegalArgumentException", message);
}

std::string to_string(JNIEnv* env, jstring value) {
    if (value == nullptr) {
        return {};
    }
    const char* chars = env->GetStringUTFChars(value, nullptr);
    if (chars == nullptr) {
        return {};
    }
    std::string out(chars);
    env->ReleaseStringUTFChars(value, chars);
    return out;
}

std::vector<std::string> to_string_vector(JNIEnv* env, jobjectArray values) {
    std::vector<std::string> out;
    if (values == nullptr) {
        return out;
    }

    const jsize length = env->GetArrayLength(values);
    out.reserve(static_cast<size_t>(length));
    for (jsize i = 0; i < length; ++i) {
        auto item = static_cast<jstring>(env->GetObjectArrayElement(values, i));
        if (env->ExceptionCheck()) {
            return {};
        }
        if (item != nullptr) {
            out.push_back(to_string(env, item));
            env->DeleteLocalRef(item);
        }
    }
    return out;
}

NativeSession* require_session(JNIEnv* env, jlong handle) {
    if (handle == 0) {
        throw_illegal_state(env, "MobileFineTuner native session is closed");
        return nullptr;
    }
    return reinterpret_cast<NativeSession*>(handle);
}

template <typename Body, typename Result>
Result guarded(JNIEnv* env, Body&& body, Result fallback) {
    try {
        return body();
    } catch (const std::invalid_argument& e) {
        throw_illegal_argument(env, e.what());
    } catch (const std::exception& e) {
        throw_illegal_state(env, e.what());
    } catch (...) {
        throw_illegal_state(env, "Unknown native MobileFineTuner error");
    }
    return fallback;
}

template <typename Body>
void guarded_void(JNIEnv* env, Body&& body) {
    try {
        body();
    } catch (const std::invalid_argument& e) {
        throw_illegal_argument(env, e.what());
    } catch (const std::exception& e) {
        throw_illegal_state(env, e.what());
    } catch (...) {
        throw_illegal_state(env, "Unknown native MobileFineTuner error");
    }
}

std::vector<jint> copy_int_array(JNIEnv* env, jintArray array, int64_t count, const char* name) {
    if (array == nullptr) {
        throw_illegal_argument(env, std::string(name) + " must not be null");
        return {};
    }
    if (env->GetArrayLength(array) != count) {
        throw_illegal_argument(env, std::string(name) + " has unexpected length");
        return {};
    }
    std::vector<jint> values(static_cast<size_t>(count));
    env->GetIntArrayRegion(array, 0, static_cast<jsize>(count), values.data());
    return values;
}

std::vector<jfloat> copy_float_array(JNIEnv* env, jfloatArray array, int64_t count, const char* name) {
    if (array == nullptr) {
        throw_illegal_argument(env, std::string(name) + " must not be null");
        return {};
    }
    if (env->GetArrayLength(array) != count) {
        throw_illegal_argument(env, std::string(name) + " has unexpected length");
        return {};
    }
    std::vector<jfloat> values(static_cast<size_t>(count));
    env->GetFloatArrayRegion(array, 0, static_cast<jsize>(count), values.data());
    return values;
}

jdoubleArray make_double_array(JNIEnv* env, const jdouble* values, jsize count) {
    jdoubleArray out = env->NewDoubleArray(count);
    if (out == nullptr) {
        return nullptr;
    }
    env->SetDoubleArrayRegion(out, 0, count, values);
    return out;
}

void fill_parameters(ops::AutoModelForCausalLM& model, float value) {
    for (const auto& param : model.parameters()) {
        if (!param || param->dtype() != ops::kFloat32) {
            continue;
        }
        float* data = param->data<float>();
        for (int64_t i = 0; i < param->numel(); ++i) {
            data[i] = value;
        }
    }
}

}  // namespace

extern "C" JNIEXPORT jstring JNICALL
Java_com_mobilefinetuner_sdk_MobileFineTuner_nativeBuildInfo(JNIEnv* env, jclass) {
    return env->NewStringUTF("MobileFineTuner Android SDK JNI: AutoModelForCausalLM + AutoTrainer");
}

extern "C" JNIEXPORT jdoubleArray JNICALL
Java_com_mobilefinetuner_sdk_MobileFineTuner_nativeSelfTest(
        JNIEnv* env,
        jclass,
        jstring working_dir) {
    return guarded(env, [&]() -> jdoubleArray {
        const std::string working_dir_string = to_string(env, working_dir);
        if (working_dir_string.empty()) {
            throw std::invalid_argument("workingDir must not be empty");
        }

        namespace fs = std::filesystem;
        const fs::path root = fs::path(working_dir_string) / "mft_sdk_native_self_test";
        fs::create_directories(root);
        {
            std::ofstream config(root / "config.json");
            if (!config.is_open()) {
                throw std::runtime_error("Failed to create self-test config.json");
            }
            config << R"({"model_type":"gpt2","vocab_size":16,"n_positions":8,"n_embd":8,"n_layer":1,"n_head":2})";
        }

        const auto t0 = std::chrono::steady_clock::now();

        ops::AutoModelLoadOptions options;
        options.load_weights = false;
        options.verbose = false;

        auto model = ops::AutoModelForCausalLM::from_pretrained(root.string(), options);
        fill_parameters(*model, 0.01f);

        ops::AutoLoraConfig lora;
        lora.rank = 2;
        lora.alpha = 4.0f;
        lora.dropout = 0.0f;
        lora.seed = 7;
        model->init_lora(lora);

        int32_t ids[] = {1, 2, 3};
        float mask[] = {1.0f, 1.0f, 1.0f};
        int32_t labels[] = {-100, 2, 3};
        auto input_ids = std::make_shared<ops::Tensor>(
            std::vector<int64_t>{1, 3}, ids, ops::kInt32, ops::kCPU);
        auto attention_mask = std::make_shared<ops::Tensor>(
            std::vector<int64_t>{1, 3}, mask, ops::kFloat32, ops::kCPU);
        auto label_tensor = std::make_shared<ops::Tensor>(
            std::vector<int64_t>{1, 3}, labels, ops::kInt32, ops::kCPU);

        ops::AutoTrainerConfig trainer_cfg;
        trainer_cfg.learning_rate = 1e-3f;
        ops::AutoTrainer trainer(*model, trainer_cfg);
        const ops::AutoTrainStepResult step =
            trainer.train_step(input_ids, attention_mask, label_tensor);

        const auto t1 = std::chrono::steady_clock::now();
        const double elapsed_ms =
            std::chrono::duration<double, std::milli>(t1 - t0).count();

        jdouble values[3] = {
            static_cast<jdouble>(step.loss),
            static_cast<jdouble>(step.trainable_tensor_count),
            static_cast<jdouble>(elapsed_ms)
        };
        return make_double_array(env, values, 3);
    }, static_cast<jdoubleArray>(nullptr));
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_mobilefinetuner_sdk_MobileFineTuner_nativeCreate(
        JNIEnv* env,
        jclass,
        jstring model_dir,
        jboolean load_weights) {
    return guarded(env, [&]() -> jlong {
        const std::string model_dir_string = to_string(env, model_dir);
        if (model_dir_string.empty()) {
            throw std::invalid_argument("modelDir must not be empty");
        }

        ops::AutoModelLoadOptions options;
        options.load_weights = (load_weights == JNI_TRUE);
        options.verbose = false;

        auto session = std::make_unique<NativeSession>();
        session->model = ops::AutoModelForCausalLM::from_pretrained(model_dir_string, options);
        return reinterpret_cast<jlong>(session.release());
    }, static_cast<jlong>(0));
}

extern "C" JNIEXPORT void JNICALL
Java_com_mobilefinetuner_sdk_MobileFineTuner_nativeInitLora(
        JNIEnv* env,
        jclass,
        jlong handle,
        jint rank,
        jfloat alpha,
        jfloat dropout,
        jlong seed,
        jobjectArray target_modules) {
    guarded_void(env, [&]() {
        NativeSession* session = require_session(env, handle);
        if (session == nullptr) {
            return;
        }
        if (!session->model) {
            throw std::runtime_error("Model is not initialized");
        }

        ops::AutoLoraConfig config;
        config.rank = rank;
        config.alpha = alpha;
        config.dropout = dropout;
        config.seed = static_cast<uint64_t>(seed);
        config.target_modules = to_string_vector(env, target_modules);
        if (env->ExceptionCheck()) {
            return;
        }
        session->model->init_lora(config);
    });
}

extern "C" JNIEXPORT void JNICALL
Java_com_mobilefinetuner_sdk_MobileFineTuner_nativeCreateTrainer(
        JNIEnv* env,
        jclass,
        jlong handle,
        jfloat learning_rate,
        jfloat weight_decay,
        jfloat max_grad_norm,
        jint ignore_index) {
    guarded_void(env, [&]() {
        NativeSession* session = require_session(env, handle);
        if (session == nullptr) {
            return;
        }
        if (!session->model) {
            throw std::runtime_error("Model is not initialized");
        }

        ops::AutoTrainerConfig config;
        config.learning_rate = learning_rate;
        config.weight_decay = weight_decay;
        config.max_grad_norm = max_grad_norm;
        config.ignore_index = ignore_index;
        session->trainer = std::make_unique<ops::AutoTrainer>(*session->model, config);
    });
}

extern "C" JNIEXPORT jdoubleArray JNICALL
Java_com_mobilefinetuner_sdk_MobileFineTuner_nativeTrainStep(
        JNIEnv* env,
        jclass,
        jlong handle,
        jintArray input_ids,
        jfloatArray attention_mask,
        jintArray labels,
        jint batch_size,
        jint sequence_length) {
    return guarded(env, [&]() -> jdoubleArray {
        NativeSession* session = require_session(env, handle);
        if (session == nullptr) {
            return nullptr;
        }
        if (!session->trainer) {
            throw std::runtime_error("Trainer is not initialized; call createTrainer first");
        }
        if (batch_size <= 0 || sequence_length <= 1) {
            throw std::invalid_argument("batchSize must be positive and sequenceLength must be > 1");
        }

        const int64_t count = static_cast<int64_t>(batch_size) * sequence_length;
        auto ids = copy_int_array(env, input_ids, count, "inputIds");
        if (env->ExceptionCheck()) {
            return nullptr;
        }
        auto mask = copy_float_array(env, attention_mask, count, "attentionMask");
        if (env->ExceptionCheck()) {
            return nullptr;
        }
        auto y = copy_int_array(env, labels, count, "labels");
        if (env->ExceptionCheck()) {
            return nullptr;
        }

        const std::vector<int64_t> shape{
            static_cast<int64_t>(batch_size),
            static_cast<int64_t>(sequence_length)
        };
        auto input_tensor = std::make_shared<ops::Tensor>(
            shape, ids.data(), ops::kInt32, ops::kCPU);
        auto mask_tensor = std::make_shared<ops::Tensor>(
            shape, mask.data(), ops::kFloat32, ops::kCPU);
        auto label_tensor = std::make_shared<ops::Tensor>(
            shape, y.data(), ops::kInt32, ops::kCPU);

        const ops::AutoTrainStepResult result =
            session->trainer->train_step(input_tensor, mask_tensor, label_tensor);

        jdouble values[2] = {
            static_cast<jdouble>(result.loss),
            static_cast<jdouble>(result.trainable_tensor_count)
        };
        return make_double_array(env, values, 2);
    }, static_cast<jdoubleArray>(nullptr));
}

extern "C" JNIEXPORT jint JNICALL
Java_com_mobilefinetuner_sdk_MobileFineTuner_nativeTrainableTensorCount(
        JNIEnv* env,
        jclass,
        jlong handle) {
    return guarded(env, [&]() -> jint {
        NativeSession* session = require_session(env, handle);
        if (session == nullptr) {
            return 0;
        }
        if (!session->model) {
            throw std::runtime_error("Model is not initialized");
        }
        return static_cast<jint>(session->model->trainable_parameters().size());
    }, static_cast<jint>(0));
}

extern "C" JNIEXPORT void JNICALL
Java_com_mobilefinetuner_sdk_MobileFineTuner_nativeClose(JNIEnv* env, jclass, jlong handle) {
    guarded_void(env, [&]() {
        delete reinterpret_cast<NativeSession*>(handle);
    });
}
