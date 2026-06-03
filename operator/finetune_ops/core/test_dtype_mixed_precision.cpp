#include "finetune_ops/core/ops.h"
#include "finetune_ops/core/lm_loss.h"
#include "finetune_ops/core/tensor.h"
#include "finetune_ops/graph/safetensors_loader.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

using namespace ops;

namespace {

void require_close(float got, float expected, float tol, const std::string& msg) {
    if (std::fabs(got - expected) > tol) {
        throw std::runtime_error(msg + ": got=" + std::to_string(got) +
                                 " expected=" + std::to_string(expected));
    }
}

TensorPtr make_tensor(const std::vector<int64_t>& shape, const std::vector<float>& values) {
    auto t = std::make_shared<Tensor>(shape, kFloat32, kCPU);
    if (t->numel() != static_cast<int64_t>(values.size())) {
        throw std::runtime_error("make_tensor: value count mismatch");
    }
    std::memcpy(t->data<float>(), values.data(), values.size() * sizeof(float));
    return t;
}

void write_u64(std::ofstream& out, uint64_t value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

void append_u16(std::vector<char>& bytes, uint16_t value) {
    char raw[sizeof(uint16_t)];
    std::memcpy(raw, &value, sizeof(uint16_t));
    bytes.insert(bytes.end(), raw, raw + sizeof(uint16_t));
}

void append_f32(std::vector<char>& bytes, float value) {
    char raw[sizeof(float)];
    std::memcpy(raw, &value, sizeof(float));
    bytes.insert(bytes.end(), raw, raw + sizeof(float));
}

std::string write_test_safetensors() {
    const std::string path = "/tmp/mf_test_dtype_mixed_precision.safetensors";
    std::vector<char> data;

    const std::vector<float> bf16_values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    for (float v : bf16_values) append_u16(data, float32_to_bf16_bits(v));
    const size_t f16_start = data.size();
    const std::vector<float> f16_values = {-1.0f, -2.0f, 0.5f, 1.5f, 2.5f, 3.5f};
    for (float v : f16_values) append_u16(data, float32_to_fp16_bits(v));
    const size_t f32_start = data.size();
    const std::vector<float> f32_values = {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
    for (float v : f32_values) append_f32(data, v);

    const std::string header =
        "{\"bf16.weight\":{\"dtype\":\"BF16\",\"shape\":[2,3],\"data_offsets\":[0," +
        std::to_string(f16_start) + "]}," +
        "\"f16.weight\":{\"dtype\":\"F16\",\"shape\":[2,3],\"data_offsets\":[" +
        std::to_string(f16_start) + "," + std::to_string(f32_start) + "]}," +
        "\"f32.weight\":{\"dtype\":\"F32\",\"shape\":[2,3],\"data_offsets\":[" +
        std::to_string(f32_start) + "," + std::to_string(data.size()) + "]}}";

    std::ofstream out(path, std::ios::binary);
    if (!out.is_open()) throw std::runtime_error("failed to create test safetensors");
    write_u64(out, static_cast<uint64_t>(header.size()));
    out.write(header.data(), static_cast<std::streamsize>(header.size()));
    out.write(data.data(), static_cast<std::streamsize>(data.size()));
    return path;
}

std::string shape_json(const std::vector<int64_t>& shape) {
    std::ostringstream oss;
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i != 0) oss << ",";
        oss << shape[i];
    }
    return oss.str();
}

void write_named_f32_safetensors(const std::string& path,
                                 const std::string& tensor_name,
                                 const std::vector<int64_t>& shape,
                                 const std::vector<float>& values) {
    int64_t numel = 1;
    for (int64_t dim : shape) numel *= dim;
    if (numel != static_cast<int64_t>(values.size())) {
        throw std::runtime_error("write_named_f32_safetensors: value count mismatch");
    }

    std::vector<char> data;
    for (float v : values) append_f32(data, v);

    const std::string header =
        "{\"" + tensor_name + "\":{\"dtype\":\"F32\",\"shape\":[" +
        shape_json(shape) + "],\"data_offsets\":[0," +
        std::to_string(data.size()) + "]}}";

    std::ofstream out(path, std::ios::binary);
    if (!out.is_open()) throw std::runtime_error("failed to create named safetensors");
    write_u64(out, static_cast<uint64_t>(header.size()));
    out.write(header.data(), static_cast<std::streamsize>(header.size()));
    out.write(data.data(), static_cast<std::streamsize>(data.size()));
}

void test_cast_roundtrip() {
    auto x = make_tensor({4}, {1.0f, -2.25f, 3.5f, 128.0f});
    auto bf16 = cast(x, kBFloat16);
    if (bf16->dtype() != kBFloat16) throw std::runtime_error("BF16 cast dtype mismatch");
    auto back = cast(bf16, kFloat32);
    for (int64_t i = 0; i < x->numel(); ++i) {
        require_close(back->data<float>()[i], x->data<float>()[i], 0.03f, "BF16 roundtrip");
    }

    auto fp16 = cast(x, kFloat16);
    if (fp16->dtype() != kFloat16) throw std::runtime_error("FP16 cast dtype mismatch");
    auto fp16_back = cast(fp16, kFloat32);
    for (int64_t i = 0; i < x->numel(); ++i) {
        require_close(fp16_back->data<float>()[i], x->data<float>()[i], 0.02f, "FP16 roundtrip");
    }
}

void test_mixed_matmul_forward_backward() {
    auto a = make_tensor({2, 3}, {1.0f, -2.0f, 3.0f, 0.5f, 4.0f, -1.0f});
    auto b_fp32 = make_tensor({3, 4}, {0.25f, 1.0f, -1.5f, 2.0f,
                                      3.0f, -0.5f, 0.75f, -2.0f,
                                      1.25f, 2.5f, -3.0f, 0.5f});
    auto b_bf16 = cast(b_fp32, kBFloat16);
    auto b_quant = cast(b_bf16, kFloat32);

    auto c = matmul(a, b_bf16);
    auto cref = matmul(a, b_quant);
    for (int64_t i = 0; i < c->numel(); ++i) {
        require_close(c->data<float>()[i], cref->data<float>()[i], 1e-5f, "mixed matmul forward");
    }

    auto bt_fp32 = make_tensor({4, 3}, {0.25f, 3.0f, 1.25f,
                                       1.0f, -0.5f, 2.5f,
                                       -1.5f, 0.75f, -3.0f,
                                       2.0f, -2.0f, 0.5f});
    auto bt_bf16 = cast(bt_fp32, kBFloat16);
    auto rhs = matmul_rhs_T(a, bt_bf16);
    auto rhs_ref = matmul_rhs_T(a, cast(bt_bf16, kFloat32));
    for (int64_t i = 0; i < rhs->numel(); ++i) {
        require_close(rhs->data<float>()[i], rhs_ref->data<float>()[i], 1e-5f, "mixed matmul_rhs_T forward");
    }

    auto a_train = make_tensor({2, 3}, {1.0f, -2.0f, 3.0f, 0.5f, 4.0f, -1.0f});
    a_train->set_requires_grad(true);
    auto y = matmul(a_train, b_bf16);
    auto loss = sum(y);
    loss->backward();
    const float* grad = a_train->grad()->data<float>();
    for (int64_t row = 0; row < 2; ++row) {
        for (int64_t k = 0; k < 3; ++k) {
            float expected = 0.0f;
            for (int64_t j = 0; j < 4; ++j) {
                expected += b_quant->data<float>()[k * 4 + j];
            }
            require_close(grad[row * 3 + k], expected, 1e-5f, "mixed matmul grad_input");
        }
    }
    if (b_bf16->grad()) {
        throw std::runtime_error("frozen BF16 weight unexpectedly received grad");
    }
}

void test_streaming_lm_loss_bf16_weight() {
    auto hidden = make_tensor({1, 3, 4}, {0.1f, -0.2f, 0.3f, 0.4f,
                                         -0.5f, 0.6f, 0.7f, -0.8f,
                                         0.9f, -1.0f, 1.1f, -1.2f});
    hidden->set_requires_grad(true);
    auto weight_fp32 = make_tensor({5, 4}, {0.2f, -0.1f, 0.4f, 0.7f,
                                           -0.3f, 0.8f, -0.6f, 0.5f,
                                           0.9f, -0.2f, 0.1f, -0.4f,
                                           0.3f, 0.6f, -0.7f, 0.2f,
                                           -0.8f, 0.5f, 0.4f, -0.1f});
    auto weight_bf16 = cast(weight_fp32, kBFloat16);
    auto weight_quant = cast(weight_bf16, kFloat32);

    auto labels = std::make_shared<Tensor>(std::vector<int64_t>{1, 3}, kInt32, kCPU);
    int32_t* y = labels->data<int32_t>();
    y[0] = 0;
    y[1] = 2;
    y[2] = 4;

    auto loss = streaming_lm_cross_entropy(hidden, weight_bf16, labels, -100, "mean");
    auto ref = streaming_lm_cross_entropy(hidden, weight_quant, labels, -100, "mean");
    require_close(loss->data<float>()[0], ref->data<float>()[0], 1e-5f, "BF16 streaming LM loss");
    loss->backward();
    if (!hidden->grad()) {
        throw std::runtime_error("BF16 streaming LM loss did not produce hidden grad");
    }
    if (weight_bf16->grad()) {
        throw std::runtime_error("frozen BF16 LM head unexpectedly received grad");
    }
}

void test_safetensors_low_precision_policy() {
    const std::string path = write_test_safetensors();
    SafeTensorsReader reader(path);
    reader.parse_header();

    std::unordered_map<std::string, std::string> mapping = {
        {"bf16.keep", "bf16.weight"},
        {"f16.promote", "f16.weight"},
        {"f32.keep", "f32.weight"}
    };

    SafeTensorsLoadOptions opts;
    opts.verbose = false;
    opts.transpose_linear = false;
    opts.auto_promote_fp16 = true;
    opts.preserve_low_precision_key_substrings = {"bf16"};
    auto tensors = reader.load_tensors_mapped(mapping, opts);
    if (tensors.at("bf16.keep")->dtype() != kBFloat16) {
        throw std::runtime_error("BF16 preserve policy failed");
    }
    if (tensors.at("f16.promote")->dtype() != kFloat32) {
        throw std::runtime_error("F16 promotion policy failed");
    }
    if (tensors.at("f32.keep")->dtype() != kFloat32) {
        throw std::runtime_error("F32 dtype changed unexpectedly");
    }
    auto bf16_as_fp32 = cast(tensors.at("bf16.keep"), kFloat32);
    require_close(bf16_as_fp32->data<float>()[5], 6.0f, 1e-5f, "BF16 safetensors value");

    SafeTensorsReader reader_t(path);
    reader_t.parse_header();
    SafeTensorsLoadOptions transpose_opts;
    transpose_opts.verbose = false;
    transpose_opts.transpose_linear = true;
    transpose_opts.auto_promote_fp16 = false;
    auto transposed = reader_t.load_tensors_mapped({{"bf16.transposed", "bf16.weight"}}, transpose_opts)
                          .at("bf16.transposed");
    if (transposed->dtype() != kBFloat16) throw std::runtime_error("BF16 transpose dtype changed");
    if (transposed->shape() != std::vector<int64_t>({3, 2})) {
        throw std::runtime_error("BF16 transpose shape mismatch");
    }
    auto transposed_fp32 = cast(transposed, kFloat32);
    const std::vector<float> expected = {1.0f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f};
    for (size_t i = 0; i < expected.size(); ++i) {
        require_close(transposed_fp32->data<float>()[i], expected[i], 1e-5f, "BF16 transpose value");
    }
}

void test_sharded_safetensors_model_reader() {
    namespace fs = std::filesystem;

    const fs::path dir = fs::temp_directory_path() / "mf_test_safetensors_sharded";
    fs::remove_all(dir);
    fs::create_directories(dir);

    write_named_f32_safetensors((dir / "model-00001-of-00002.safetensors").string(),
                                "a.weight",
                                {2},
                                {1.0f, 2.0f});
    write_named_f32_safetensors((dir / "model-00002-of-00002.safetensors").string(),
                                "b.weight",
                                {2},
                                {3.0f, 4.0f});

    std::ofstream index(dir / "model.safetensors.index.json");
    if (!index.is_open()) throw std::runtime_error("failed to create safetensors index");
    index << "{\"metadata\":{\"total_size\":16},\"weight_map\":{"
          << "\"a.weight\":\"model-00001-of-00002.safetensors\","
          << "\"b.weight\":\"model-00002-of-00002.safetensors\"}}";
    index.close();

    SafeTensorsModelReader reader(dir.string());
    reader.parse_headers();
    if (reader.files().size() != 2) {
        throw std::runtime_error("sharded reader did not resolve both shards");
    }
    const auto names = reader.get_tensor_names();
    if (names != std::vector<std::string>({"a.weight", "b.weight"})) {
        throw std::runtime_error("sharded reader tensor names mismatch");
    }

    SafeTensorsLoadOptions opts;
    opts.verbose = false;
    opts.transpose_linear = false;
    auto tensors = reader.load_tensors_mapped({{"a", "a.weight"}, {"b", "b.weight"}}, opts);

    require_close(tensors.at("a")->data<float>()[0], 1.0f, 1e-6f, "sharded tensor a[0]");
    require_close(tensors.at("a")->data<float>()[1], 2.0f, 1e-6f, "sharded tensor a[1]");
    require_close(tensors.at("b")->data<float>()[0], 3.0f, 1e-6f, "sharded tensor b[0]");
    require_close(tensors.at("b")->data<float>()[1], 4.0f, 1e-6f, "sharded tensor b[1]");
}

} // namespace

int main() {
    test_cast_roundtrip();
    test_mixed_matmul_forward_backward();
    test_streaming_lm_loss_bf16_weight();
    test_safetensors_low_precision_policy();
    test_sharded_safetensors_model_reader();
    std::cout << "[PASS] dtype mixed precision tests" << std::endl;
    return 0;
}
