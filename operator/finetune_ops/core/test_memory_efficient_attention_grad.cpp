#include "memory_efficient_attention.h"
#include "autograd_engine.h"
#include "ops.h"

#include <cmath>
#include <iostream>

using namespace ops;

namespace {

float rel_diff(const TensorPtr& a, const TensorPtr& b) {
    const float* pa = a->data<float>();
    const float* pb = b->data<float>();
    double nd = 0.0;
    double nb = 0.0;
    for (int64_t i = 0; i < a->numel(); ++i) {
        double da = static_cast<double>(pa[i]);
        double db = static_cast<double>(pb[i]);
        nd += (da - db) * (da - db);
        nb += db * db;
    }
    if (nb == 0.0) {
        return static_cast<float>(std::sqrt(nd));
    }
    return static_cast<float>(std::sqrt(nd / nb));
}

void fill_tensor(const TensorPtr& tensor, float scale, int mod, int offset) {
    float* data = tensor->data<float>();
    for (int64_t i = 0; i < tensor->numel(); ++i) {
        data[i] = scale * static_cast<float>((i % mod) - offset);
    }
}

TensorPtr make_tensor(const std::vector<int64_t>& shape, float scale, int mod, int offset) {
    auto tensor = zeros(shape, kFloat32, kCPU);
    fill_tensor(tensor, scale, mod, offset);
    tensor->set_requires_grad(true);
    return tensor;
}

TensorPtr make_tensor_no_grad(const std::vector<int64_t>& shape, float scale, int mod, int offset) {
    auto tensor = zeros(shape, kFloat32, kCPU);
    fill_tensor(tensor, scale, mod, offset);
    return tensor;
}

float l2_norm(const TensorPtr& tensor) {
    const float* data = tensor->data<float>();
    double norm = 0.0;
    for (int64_t i = 0; i < tensor->numel(); ++i) {
        norm += static_cast<double>(data[i]) * static_cast<double>(data[i]);
    }
    return static_cast<float>(std::sqrt(norm));
}

}  // namespace

int main() {
    const int64_t B = 1;
    const int64_t H = 2;
    const int64_t S = 4;
    const int64_t D = 3;
    const float scale = 1.0f / std::sqrt(static_cast<float>(D));

    auto q_fast = make_tensor({B, H, S, D}, 0.07f, 11, 5);
    auto k_fast = make_tensor({B, H, S, D}, 0.05f, 9, 4);
    auto v_fast = make_tensor({B, H, S, D}, 0.03f, 7, 3);
    auto q_ref = make_tensor_no_grad({B, H, S, D}, 0.07f, 11, 5);
    auto k_ref = make_tensor_no_grad({B, H, S, D}, 0.05f, 9, 4);
    auto v_ref = make_tensor_no_grad({B, H, S, D}, 0.03f, 7, 3);

    auto causal_mask = create_causal_mask(static_cast<int>(S), kFloat32, kCPU);
    MemoryEfficientAttentionConfig config;
    config.use_causal_mask = true;
    config.scale = scale;

    auto out_fast = memory_efficient_attention(q_fast, k_fast, v_fast, causal_mask, config);

    auto scores_ref = mul(matmul(q_ref, transpose(k_ref, 2, 3)), scale);
    scores_ref = add(scores_ref, causal_mask);
    auto probs_ref = softmax(scores_ref, -1);
    auto out_ref = matmul(probs_ref, v_ref);

    auto grad_out_fast = zeros(out_fast->shape(), kFloat32, kCPU);
    fill_tensor(grad_out_fast, 0.02f, 13, 6);

    {
        using namespace ops::autograd;
        Engine::instance().run_backward({out_fast}, {grad_out_fast});
    }

    if (!q_fast->grad() || !k_fast->grad() || !v_fast->grad()) {
        std::cerr << "[MemEffAttn] missing grads: "
                  << "fast(q=" << (q_fast->grad() ? "y" : "n")
                  << ",k=" << (k_fast->grad() ? "y" : "n")
                  << ",v=" << (v_fast->grad() ? "y" : "n")
                  << ")" << std::endl;
        return 1;
    }

    float out_diff = rel_diff(out_fast, out_ref);
    float dq_norm = l2_norm(q_fast->grad());
    float dk_norm = l2_norm(k_fast->grad());
    float dv_norm = l2_norm(v_fast->grad());

    std::cout << "[MemEffAttn] out=" << out_diff
              << " dq_norm=" << dq_norm
              << " dk_norm=" << dk_norm
              << " dv_norm=" << dv_norm << std::endl;

    bool ok = out_diff < 1e-6f && dq_norm > 1e-6f && dk_norm > 1e-6f && dv_norm > 1e-6f;
    std::cout << (ok ? "[PASS]" : "[FAIL]") << std::endl;
    return ok ? 0 : 1;
}
