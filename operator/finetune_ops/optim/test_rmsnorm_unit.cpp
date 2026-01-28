#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include "../core/tensor.h"
#include "../core/ops.h"
#include "../core/autograd_engine.h"

using namespace ops;

static TensorPtr make_random(const std::vector<int64_t>& shape, float scale = 1.0f, uint64_t seed = 42) {
    auto t = zeros(shape, DType::kFloat32, kCPU);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-scale, scale);
    float* data = t->data<float>();
    for (int64_t i = 0; i < t->numel(); ++i) data[i] = dist(rng);
    return t;
}

static float l2(const TensorPtr& t) {
    const float* p = t->data<float>();
    double s = 0.0;
    for (int64_t i = 0; i < t->numel(); ++i) s += static_cast<double>(p[i]) * static_cast<double>(p[i]);
    return static_cast<float>(std::sqrt(s));
}

static float rel_diff(const TensorPtr& a, const TensorPtr& b) {
    auto da = a->data<float>();
    auto db = b->data<float>();
    double num = 0.0, den = 0.0;
    for (int64_t i = 0; i < a->numel(); ++i) {
        double d = static_cast<double>(da[i]) - static_cast<double>(db[i]);
        num += d * d;
        den += static_cast<double>(db[i]) * static_cast<double>(db[i]);
    }
    return static_cast<float>(std::sqrt(num) / (std::sqrt(den) + 1e-12));
}

int main() {
    // Small but non-trivial ND shapes to test broadcasting and accumulation
    std::vector<std::vector<int64_t>> shapes = {
        {16},            // 1D
        {3, 8},         // 2D
        {2, 3, 8},      // 3D
        {1, 4, 3, 8},   // 4D (matches [B,H,S,D]-like structure)
    };

    bool all_ok = true;
    const float eps = 1e-5f;

    for (const auto& shape : shapes) {
        // Generate inputs
        auto x = make_random(shape, 1.0f, 123);
        auto w = make_random({shape.back()}, 0.1f, 456);  // weight only on last dim
        x->set_requires_grad(true);
        w->set_requires_grad(true);

        // Forward RMSNorm: y = x * inv_rms * (1 + w)
        auto y = rms_norm(x, w, eps);

        // Random upstream gradient for y
        auto gy = make_random(shape, 1.0f, 789);

        // Backward analytic
        using namespace ops::autograd;
        Engine::instance().run_backward({y}, {gy});
        auto gx_analytic = x->grad();
        auto gw_analytic = w->grad();

        // Numerical gradient for x
        const float h = 1e-3f;
        auto gx_num = zeros(shape, DType::kFloat32, kCPU);
        float* gx_num_data = gx_num->data<float>();

        // Snapshot original x
        std::vector<float> x_orig(x->numel());
        std::copy(x->data<float>(), x->data<float>() + x->numel(), x_orig.begin());

        // Compute <gy, dy/dx_i> via finite differences
        for (int64_t i = 0; i < x->numel(); ++i) {
            float* x_data = x->data<float>();
            x_data[i] = x_orig[i] + h;
            auto y_pos = rms_norm(x, w, eps);
            x_data[i] = x_orig[i] - h;
            auto y_neg = rms_norm(x, w, eps);
            // restore
            x_data[i] = x_orig[i];

            // dy ≈ (y_pos - y_neg) / (2h)
            auto dy = sub(y_pos, y_neg);
            auto dy_scaled = mul(dy, 1.0f / (2.0f * h));
            // <gy, dy> reduced to scalar
            auto prod = mul(dy_scaled, gy);
            auto s = prod;
            while (s->ndim() > 1) {
                s = sum(s, -1, false);
            }
            s = sum(s, -1, false);
            gx_num_data[i] = s->data<float>()[0]; 
        }

        float rd_x = rel_diff(gx_analytic, gx_num);

        // Numerical gradient for w (last dimension only)
        auto gw_num = zeros({shape.back()}, DType::kFloat32, kCPU);
        float* gw_num_data = gw_num->data<float>();
        std::vector<float> w_orig(w->numel());
        std::copy(w->data<float>(), w->data<float>() + w->numel(), w_orig.begin());
        for (int64_t i = 0; i < w->numel(); ++i) {
            float* w_data = w->data<float>();
            w_data[i] = w_orig[i] + h;
            auto y_pos = rms_norm(x, w, eps);
            w_data[i] = w_orig[i] - h;
            auto y_neg = rms_norm(x, w, eps);
            // restore
            w_data[i] = w_orig[i];

            auto dy = sub(y_pos, y_neg);
            auto dy_scaled = mul(dy, 1.0f / (2.0f * h));
            auto prod = mul(dy_scaled, gy);
            auto s = prod;
            while (s->ndim() > 1) {
                s = sum(s, -1, false);
            }
            s = sum(s, -1, false);
            gw_num_data[i] = s->data<float>()[0];
        }

        float rd_w = rel_diff(gw_analytic, gw_num);

        std::cout << "[T4.RMSNorm] shape=[";
        for (size_t i = 0; i < shape.size(); ++i) {
            std::cout << shape[i] << (i + 1 < shape.size() ? "," : "");
        }
        std::cout << "] rel_diff gx=" << rd_x << " gw=" << rd_w << std::endl;

        bool ok = (rd_x < 5e-3f) && (rd_w < 5e-3f);
        all_ok = all_ok && ok;
    }

    if (all_ok) {
        std::cout << "[PASS]" << std::endl;
        return 0;
    }
    std::cout << "[FAIL]" << std::endl;
    return 1;
}


