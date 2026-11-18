#include "qwen35_ops.h"
#include "autograd_engine.h"
#include "backward_functions.h"
#include "ops.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace ops {
namespace {
void fp32(const TensorPtr& t) {
    if (!t || t->dtype() != kFloat32 || !t->device().is_cpu())
        throw std::invalid_argument("Qwen3.5 kernels require CPU FP32 tensors");
}
void attach(const TensorPtr& out, const std::vector<TensorPtr>& inputs,
            BackwardFunctionPtr backward) {
    if (!autograd::Engine::instance().is_enabled())
        return;
    if (std::none_of(inputs.begin(), inputs.end(),
                     [](const TensorPtr& t) { return t->requires_grad(); }))
        return;
#ifdef USE_NEW_AUTOGRAD_ENGINE
    out->set_requires_grad(true);
    autograd::Engine::instance().register_node(out, inputs, std::move(backward));
#else
    (void)out;
    (void)backward;
    throw std::logic_error("Qwen3.5 training requires the topological autograd engine");
#endif
}
float sigmoid_value(float x) {
    return x >= 0 ? 1.0f / (1.0f + std::exp(-x)) : std::exp(x) / (1.0f + std::exp(x));
}
float softplus_value(float x) { return x > 20.0f ? x : std::log1p(std::exp(x)); }

struct ConvBackward : BackwardFunction {
    TensorPtr x, w;
    ConvBackward(TensorPtr xx, TensorPtr ww) : x(xx), w(ww) {}
    std::vector<TensorPtr> apply(const TensorPtr& go) override {
        fp32(go);
        if (go->shape() != x->shape())
            throw std::invalid_argument("Conv gradient shape");
        auto dx = zeros(x->shape()), dw = zeros(w->shape());
        const auto B = x->shape()[0], S = x->shape()[1], C = x->shape()[2], K = w->shape()[2];
        for (int64_t b = 0; b < B; ++b)
            for (int64_t t = 0; t < S; ++t)
                for (int64_t c = 0; c < C; ++c)
                    for (int64_t j = 0; j < K; ++j) {
                        const auto source = t - K + 1 + j;
                        if (source < 0)
                            continue;
                        const auto xi = (b * S + source) * C + c;
                        const float g = go->data<float>()[(b * S + t) * C + c];
                        dx->data<float>()[xi] += g * w->data<float>()[c * K + j];
                        dw->data<float>()[c * K + j] += g * x->data<float>()[xi];
                    }
        return {dx, dw};
    }
};

TensorPtr rotary(const TensorPtr& x, int R, float theta, bool inverse) {
    const auto shape = x->shape();
    auto out = zeros(shape);
    const int64_t S = shape[2], D = shape[3], rows = x->numel() / D;
    std::copy(x->data<float>(), x->data<float>() + x->numel(), out->data<float>());
    for (int64_t row = 0; row < rows; ++row)
        for (int d = 0; d < R / 2; ++d) {
            const float angle = float(row % S) / std::pow(theta, 2.0f * d / R);
            const float c = std::cos(angle), s = (inverse ? -1.0f : 1.0f) * std::sin(angle);
            const auto i = row * D + d, j = i + R / 2;
            out->data<float>()[i] = x->data<float>()[i] * c - x->data<float>()[j] * s;
            out->data<float>()[j] = x->data<float>()[i] * s + x->data<float>()[j] * c;
        }
    return out;
}
struct RotaryBackward : BackwardFunction {
    int R;
    float theta;
    std::vector<int64_t> shape;
    RotaryBackward(int r, float t, std::vector<int64_t> s) : R(r), theta(t), shape(std::move(s)) {}
    std::vector<TensorPtr> apply(const TensorPtr& go) override {
        fp32(go);
        if (go->shape() != shape)
            throw std::invalid_argument("Partial RoPE gradient shape");
        return {rotary(go, R, theta, true)};
    }
};

struct DeltaBackward : BackwardFunction {
    std::vector<TensorPtr> inputs;
    int64_t B, S, Hk, Hv, K, V;
    std::vector<float> q, k, qi, ki, e, beta, states;
    explicit DeltaBackward(std::vector<TensorPtr> p) : inputs(std::move(p)) {
        const auto& qs = inputs[0]->shape();
        const auto& vs = inputs[2]->shape();
        B = qs[0];
        S = qs[1];
        Hk = qs[2];
        K = qs[3];
        Hv = vs[2];
        V = vs[3];
        q.resize(inputs[0]->numel());
        k.resize(inputs[1]->numel());
        qi.resize(B * S * Hk);
        ki.resize(qi.size());
        e.resize(B * S * Hv);
        beta.resize(e.size());
        if (uint64_t(B) * Hv * uint64_t(S + 1) >
            std::numeric_limits<size_t>::max() / sizeof(float) / K / V)
            throw std::invalid_argument("Gated DeltaNet state size overflow");
        states.resize(size_t(B) * Hv * (S + 1) * K * V, 0.0f);
    }
    void normalize() {
        const float scale = 1.0f / std::sqrt(float(K));
        for (int64_t row = 0; row < B * S * Hk; ++row) {
            float qsum = 0, ksum = 0;
            for (int64_t d = 0; d < K; ++d) {
                const auto i = row * K + d;
                qsum += inputs[0]->data<float>()[i] * inputs[0]->data<float>()[i];
                ksum += inputs[1]->data<float>()[i] * inputs[1]->data<float>()[i];
            }
            qi[row] = 1.0f / std::sqrt(qsum + 1e-6f);
            ki[row] = 1.0f / std::sqrt(ksum + 1e-6f);
            for (int64_t d = 0; d < K; ++d) {
                const auto i = row * K + d;
                q[i] = inputs[0]->data<float>()[i] * qi[row] * scale;
                k[i] = inputs[1]->data<float>()[i] * ki[row];
            }
        }
        for (int64_t row = 0; row < B * S * Hv; ++row) {
            const auto h = row % Hv;
            const float g =
                -std::exp(inputs[5]->data<float>()[h]) *
                softplus_value(inputs[3]->data<float>()[row] + inputs[6]->data<float>()[h]);
            e[row] = std::exp(g);
            beta[row] = sigmoid_value(inputs[4]->data<float>()[row]);
        }
    }
    TensorPtr forward() {
        normalize();
        auto out = zeros({B, S, Hv, V});
        std::vector<float> delta(V);
        for (int64_t b = 0; b < B; ++b)
            for (int64_t h = 0; h < Hv; ++h) {
                const auto statebase = (b * Hv + h) * (S + 1) * K * V, hk = h / (Hv / Hk);
                for (int64_t t = 0; t < S; ++t) {
                    const auto row = (b * S + t) * Hv + h, qbase = ((b * S + t) * Hk + hk) * K,
                               vbase = row * V;
                    const float* prev = states.data() + statebase + t * K * V;
                    float* next = states.data() + statebase + (t + 1) * K * V;
                    for (int64_t v = 0; v < V; ++v) {
                        float memory = 0;
                        for (int64_t d = 0; d < K; ++d)
                            memory += (prev[d * V + v] * e[row]) * k[qbase + d];
                        delta[v] = (inputs[2]->data<float>()[vbase + v] - memory) * beta[row];
                    }
                    for (int64_t d = 0; d < K; ++d)
                        for (int64_t v = 0; v < V; ++v)
                            next[d * V + v] = prev[d * V + v] * e[row] + k[qbase + d] * delta[v];
                    for (int64_t v = 0; v < V; ++v) {
                        float value = 0;
                        for (int64_t d = 0; d < K; ++d)
                            value += next[d * V + v] * q[qbase + d];
                        out->data<float>()[vbase + v] = value;
                    }
                }
            }
        return out;
    }
    std::vector<TensorPtr> apply(const TensorPtr& go) override {
        fp32(go);
        if (go->shape() != std::vector<int64_t>{B, S, Hv, V})
            throw std::invalid_argument("DeltaNet gradient shape");
        std::vector<TensorPtr> grad;
        for (const auto& p : inputs)
            grad.push_back(zeros(p->shape()));
        std::vector<float> dh(K * V), dp(K * V), delta(V), diff(V), dd(V), du(V);
        for (int64_t b = 0; b < B; ++b)
            for (int64_t h = 0; h < Hv; ++h) {
                std::fill(dh.begin(), dh.end(), 0.0f);
                const auto statebase = (b * Hv + h) * (S + 1) * K * V, hk = h / (Hv / Hk);
                for (int64_t t = S; t-- > 0;) {
                    const auto row = (b * S + t) * Hv + h, qbase = ((b * S + t) * Hk + hk) * K,
                               vbase = row * V;
                    const float* prev = states.data() + statebase + t * K * V;
                    const float* current = prev + K * V;
                    for (int64_t d = 0; d < K; ++d) {
                        float dq = 0;
                        for (int64_t v = 0; v < V; ++v) {
                            const float dy = go->data<float>()[vbase + v];
                            dq += current[d * V + v] * dy;
                            dh[d * V + v] += q[qbase + d] * dy;
                        }
                        grad[0]->data<float>()[qbase + d] += dq;
                    }
                    for (int64_t v = 0; v < V; ++v) {
                        float memory = 0, ddelta = 0;
                        for (int64_t d = 0; d < K; ++d) {
                            memory += (prev[d * V + v] * e[row]) * k[qbase + d];
                            ddelta += dh[d * V + v] * k[qbase + d];
                        }
                        diff[v] = inputs[2]->data<float>()[vbase + v] - memory;
                        delta[v] = diff[v] * beta[row];
                        dd[v] = ddelta;
                        du[v] = -beta[row] * ddelta;
                        grad[2]->data<float>()[vbase + v] += beta[row] * ddelta;
                    }
                    float de = 0, dbeta = 0;
                    for (int64_t v = 0; v < V; ++v)
                        dbeta += dd[v] * diff[v];
                    for (int64_t d = 0; d < K; ++d) {
                        float dk = 0;
                        for (int64_t v = 0; v < V; ++v) {
                            const auto i = d * V + v;
                            dk += dh[i] * delta[v] + prev[i] * e[row] * du[v];
                            dp[i] = dh[i] + k[qbase + d] * du[v];
                            de += dp[i] * prev[i];
                            dh[i] = e[row] * dp[i];
                        }
                        grad[1]->data<float>()[qbase + d] += dk;
                    }
                    const float dg = de * e[row], ea = std::exp(inputs[5]->data<float>()[h]);
                    const float pre = inputs[3]->data<float>()[row] + inputs[6]->data<float>()[h];
                    const float da = -dg * ea * sigmoid_value(pre);
                    grad[3]->data<float>()[row] += da;
                    grad[4]->data<float>()[row] += dbeta * beta[row] * (1.0f - beta[row]);
                    grad[5]->data<float>()[h] += -dg * ea * softplus_value(pre);
                    grad[6]->data<float>()[h] += da;
                }
            }
        for (int input = 0; input < 2; ++input)
            for (int64_t row = 0; row < B * S * Hk; ++row) {
                const float inv = input == 0 ? qi[row] : ki[row],
                            scale = input == 0 ? 1.0f / std::sqrt(float(K)) : 1.0f;
                float dot = 0;
                for (int64_t d = 0; d < K; ++d)
                    dot += grad[input]->data<float>()[row * K + d] *
                           inputs[input]->data<float>()[row * K + d];
                for (int64_t d = 0; d < K; ++d) {
                    const auto i = row * K + d;
                    grad[input]->data<float>()[i] =
                        inv * scale *
                        (grad[input]->data<float>()[i] -
                         inputs[input]->data<float>()[i] * inv * inv * dot);
                }
            }
        return grad;
    }
};
} // namespace

TensorPtr causal_depthwise_conv1d(const TensorPtr& x, const TensorPtr& w) {
    fp32(x);
    fp32(w);
    if (x->ndim() != 3 || w->ndim() != 3 || w->shape()[0] != x->shape()[2] || w->shape()[1] != 1 ||
        w->shape()[2] <= 0)
        throw std::invalid_argument("Causal depthwise convolution expects [B,S,C] and [C,1,K]");
    const auto B = x->shape()[0], S = x->shape()[1], C = x->shape()[2], K = w->shape()[2];
    auto out = zeros(x->shape());
    for (int64_t b = 0; b < B; ++b)
        for (int64_t t = 0; t < S; ++t)
            for (int64_t c = 0; c < C; ++c) {
                float value = 0;
                for (int64_t j = 0; j < K; ++j)
                    if (t - K + 1 + j >= 0)
                        value += x->data<float>()[(b * S + t - K + 1 + j) * C + c] *
                                 w->data<float>()[c * K + j];
                out->data<float>()[(b * S + t) * C + c] = value;
            }
    attach(out, {x, w}, std::make_shared<ConvBackward>(x, w));
    return out;
}
TensorPtr partial_rotary_embedding(const TensorPtr& x, int R, float theta) {
    fp32(x);
    if (x->ndim() != 4 || R <= 0 || R % 2 || R > x->shape()[3] || !std::isfinite(theta) ||
        theta <= 0)
        throw std::invalid_argument("Invalid partial rotary dimensions/theta");
    auto out = rotary(x, R, theta, false);
    attach(out, {x}, std::make_shared<RotaryBackward>(R, theta, x->shape()));
    return out;
}
TensorPtr gated_delta_rule(const TensorPtr& q, const TensorPtr& k, const TensorPtr& v,
                           const TensorPtr& a, const TensorPtr& b, const TensorPtr& logA,
                           const TensorPtr& dt) {
    const std::vector<TensorPtr> p = {q, k, v, a, b, logA, dt};
    for (const auto& t : p)
        fp32(t);
    if (q->ndim() != 4 || k->shape() != q->shape() || v->ndim() != 4 ||
        v->shape()[0] != q->shape()[0] || v->shape()[1] != q->shape()[1] || q->shape()[2] <= 0 ||
        v->shape()[2] % q->shape()[2] || v->shape()[2] < q->shape()[2])
        throw std::invalid_argument("Invalid Gated DeltaNet Q/K/V shapes");
    const auto B = q->shape()[0], S = q->shape()[1], H = v->shape()[2];
    if (B <= 0 || S <= 0 || q->shape()[3] <= 0 || v->shape()[3] <= 0 ||
        a->shape() != std::vector<int64_t>{B, S, H} || b->shape() != a->shape() ||
        logA->shape() != std::vector<int64_t>{H} || dt->shape() != logA->shape())
        throw std::invalid_argument("Invalid Gated DeltaNet gate/state dimensions");
    auto kernel = std::make_shared<DeltaBackward>(p);
    auto out = kernel->forward();
    attach(out, p, kernel);
    return out;
}
} // namespace ops
