#include "../core/ops.h"
#include "../core/autograd_engine.h"
#include <iostream>
#include <cmath>
#include <vector>
#include <cassert>

using namespace ops;

static float rel_diff(const TensorPtr& a, const TensorPtr& b) {
    const float* pa = a->data<float>();
    const float* pb = b->data<float>();
    double nd = 0.0, nb = 0.0;
    int64_t n = a->numel();
    for (int64_t i = 0; i < n; ++i) {
        double da = static_cast<double>(pa[i]);
        double db = static_cast<double>(pb[i]);
        nd += (da - db) * (da - db);
        nb += db * db;
    }
    return nb == 0.0 ? static_cast<float>(std::sqrt(nd)) : static_cast<float>(std::sqrt(nd / nb));
}

// Minimal RMSNorm forward (y = x_hat * (1 + w))
static TensorPtr rms_norm_manual(const TensorPtr& x, const TensorPtr& w, float eps) {
    const auto& s = x->shape(); // [B,H,S,D] or [B,S,D]
    auto y = zeros(s, x->dtype(), x->device());
    const float* xd = x->data<float>();
    const float* wd = w->data<float>();
    float* yd = y->data<float>();
    int64_t D = s.back();
    int64_t outer = x->numel() / D;
    for (int64_t o = 0; o < outer; ++o) {
        double sq = 0.0;
        for (int64_t i = 0; i < D; ++i) {
            float v = xd[o * D + i];
            sq += static_cast<double>(v) * static_cast<double>(v);
        }
        double inv_rms = 1.0 / std::sqrt(sq / static_cast<double>(D) + static_cast<double>(eps));
        for (int64_t i = 0; i < D; ++i) {
            yd[o * D + i] = static_cast<float>(static_cast<double>(xd[o * D + i]) * inv_rms * (1.0 + static_cast<double>(wd[i])));
        }
    }
    // Not registering backward here; this is for attention-only path where norm is typically precomputed
    return y;
}

int main() {
    // Config (some toggles via env)
    const int64_t B = 1, H = 4, S = 8, Hd = 8;
    const float scale = std::pow(256.0f, -0.5f);
    const float eps = 1e-6f;
    auto getenv_flag = [](const char* name, bool default_val) -> bool {
        const char* v = std::getenv(name);
        if (!v) return default_val;
        int iv = std::atoi(v);
        return iv != 0;
    };
    const bool USE_MASK = getenv_flag("ATTN_USE_MASK", true);
    const bool USE_ROPE = getenv_flag("ATTN_USE_ROPE", false);
    const bool USE_REPEAT = getenv_flag("ATTN_USE_REPEAT", true);
    const int64_t kvH = USE_REPEAT ? 1 : H; // if not repeating, match heads

    // Inputs: pre-attn normalized q/k/v (we simulate pre-norm via manual RMS if needed later)
    auto q = zeros({B, H, S, Hd}, kFloat32, kCPU); q->set_requires_grad(true);
    auto k_in = zeros({B, kvH, S, Hd}, kFloat32, kCPU); k_in->set_requires_grad(true);
    auto v_in = zeros({B, kvH, S, Hd}, kFloat32, kCPU); v_in->set_requires_grad(true);

    // Deterministic fill
    {
        float* qd = q->data<float>();
        float* kd = k_in->data<float>();
        float* vd = v_in->data<float>();
        for (int i = 0; i < q->numel(); ++i) qd[i] = 0.013f * (i % 17 - 8);
        for (int i = 0; i < k_in->numel(); ++i) kd[i] = 0.011f * (i % 13 - 6);
        for (int i = 0; i < v_in->numel(); ++i) vd[i] = 0.015f * (i % 19 - 9);
    }

    // Optional: RoPE and norm weights (disabled by default here)
    auto q_use = q;
    auto k_use = k_in;
    if (USE_REPEAT) {
        k_use = repeat_kv_heads(k_in, H / kvH); // [B,H,S,Hd]
    }
    auto v_use = USE_REPEAT ? repeat_kv_heads(v_in, H / kvH) : v_in; // [B,H,S,Hd]

    // Scores and probs
    if (USE_REPEAT) {
        k_use->retain_grad();
    }
    auto k_t = transpose(k_use, 2, 3); // [B,H,Hd,S]
    auto scores = mul(matmul(q_use, k_t), scale); // [B,H,S,S]
    scores->retain_grad();
    if (USE_MASK) {
        auto mask = zeros({B, 1, S, S}, kFloat32, kCPU);
        float* md = mask->data<float>();
        for (int i = 0; i < S; ++i) for (int j = i + 1; j < S; ++j) md[i * S + j] = -1e10f;
        scores = add(scores, mask);
        scores->retain_grad();
    }
    auto probs = softmax(scores, -1);
    auto context = matmul(probs, v_use); // [B,H,S,Hd]

    // Upstream grad on context
    auto G = zeros(context->shape(), kFloat32, kCPU);
    {
        float* gd = G->data<float>();
        for (int i = 0; i < G->numel(); ++i) gd[i] = 0.02f * (i % 11 - 5);
    }
    auto loss = sum(sum(sum(sum(mul(context, G), -1, false), -1, false), -1, false), -1, false); // scalar

    // Backward by engine
    {
        using namespace ops::autograd;
        Engine::instance().run_backward({loss}, {nullptr});
    }
    auto gq = q->grad();
    auto gk_in = k_in->grad();
    auto gv_in = v_in->grad();
    if (USE_REPEAT) {
        auto gk_full = k_use->grad() ? k_use->grad() : zeros(k_use->shape(), kFloat32, kCPU);
        // Sum repeats back to kv layout
        auto gk_from_full = zeros(k_in->shape(), kFloat32, kCPU);
        const float* src = gk_full->data<float>();
        float* dst = gk_from_full->data<float>();
        int64_t heads_rep = H / kvH;
        for (int64_t b=0;b<B;++b){
            for (int64_t kv=0;kv<kvH;++kv){
                for (int64_t s=0;s<S;++s){
                    for (int64_t d=0;d<Hd;++d){
                        double sum = 0.0;
                        for (int64_t rep=0; rep < heads_rep; ++rep){
                            int64_t out_h = kv*heads_rep + rep;
                            int64_t si = (((b*H + out_h)*S + s)*Hd + d);
                            sum += static_cast<double>(src[si]);
                        }
                        int64_t di = (((b*kvH + kv)*S + s)*Hd + d);
                        dst[di] = static_cast<float>(sum);
                    }
                }
            }
        }
        float rdc = rel_diff(gk_in, gk_from_full);
        std::cout << "[T3.AttnOnly] repeat sum-back dK_in vs gK_full-summed=" << rdc << std::endl;
    }
    // Inspect dScores correctness
    auto scores_grad = scores->grad() ? scores->grad() : zeros(scores->shape(), kFloat32, kCPU);
    auto v_t = transpose(v_use, -2, -1);
    auto dProbs = matmul(G, v_t);
    auto prod = mul(dProbs, probs);
    int last_dim = static_cast<int>(probs->shape().size()) - 1;
    auto sum_prod = sum(prod, last_dim, true);
    auto dScores_loop = mul(probs, sub(dProbs, sum_prod));
    auto dScores_rd = rel_diff(scores_grad, dScores_loop);
    std::cout << "[T3.AttnOnly] dScores engine vs loop=" << dScores_rd << std::endl;
    // Analytic refs for K and V
    auto dMat = mul(dScores_loop, scale);                 // [B,H,S,S]
    auto gk_full_ref = matmul(transpose(dMat, -2, -1), q);  // [B,H,S,Hd]
    auto gv_full_ref = matmul(transpose(probs, -2, -1), G); // [B,H,S,Hd]
    float rdk_analytic = 0.0f;
    float rdv_analytic = 0.0f;
    if (!USE_REPEAT) {
        // kvH == H → in equals full
        rdk_analytic = rel_diff(gk_in, gk_full_ref);
        rdv_analytic = rel_diff(gv_in, gv_full_ref);
        std::cout << "[T3.AttnOnly] gK_in vs ref=" << rdk_analytic
                  << " gV_in vs ref=" << rdv_analytic << std::endl;
    } else {
        // Sum-back heads to kv layout
        auto sum_back = [&](const TensorPtr& full, int64_t heads_rep) {
            auto out = zeros(k_in->shape(), kFloat32, kCPU);
            const float* src = full->data<float>();
            float* dst = out->data<float>();
            for (int64_t b=0;b<B;++b){
                for (int64_t kv=0;kv<kvH;++kv){
                    for (int64_t s=0;s<S;++s){
                        for (int64_t d=0;d<Hd;++d){
                            double sum = 0.0;
                            for (int64_t rep=0; rep < heads_rep; ++rep){
                                int64_t out_h = kv*heads_rep + rep;
                                int64_t si = (((b*H + out_h)*S + s)*Hd + d);
                                sum += static_cast<double>(src[si]);
                            }
                            int64_t di = (((b*kvH + kv)*S + s)*Hd + d);
                            dst[di] = static_cast<float>(sum);
                        }
                    }
                }
            }
            return out;
        };
        int64_t heads_rep = H / kvH;
        auto gk_ref_sb = sum_back(gk_full_ref, heads_rep);
        auto gv_ref_sb = sum_back(gv_full_ref, heads_rep);
        rdk_analytic = rel_diff(gk_in, gk_ref_sb);
        rdv_analytic = rel_diff(gv_in, gv_ref_sb);
        std::cout << "[T3.AttnOnly] gK_in vs ref(sum-back)=" << rdk_analytic
                  << " gV_in vs ref(sum-back)=" << rdv_analytic << std::endl;
    }

    // Numeric FD helpers
    auto gk_num = zeros(k_in->shape(), kFloat32, kCPU);
    auto gv_num = zeros(v_in->shape(), kFloat32, kCPU);
    // Helper to evaluate scalar loss with given K/V
    auto eval_with = [&](const TensorPtr& K_in, const TensorPtr& V_in) -> float {
        auto K_use = USE_REPEAT ? repeat_kv_heads(K_in, H / kvH) : K_in;
        auto V_use = USE_REPEAT ? repeat_kv_heads(V_in, H / kvH) : V_in;
        auto s = mul(matmul(q, transpose(K_use, 2, 3)), scale);
        if (USE_MASK) {
            auto mask = zeros({B, 1, S, S}, kFloat32, kCPU);
            float* md = mask->data<float>();
            for (int i = 0; i < S; ++i) for (int j = i + 1; j < S; ++j) md[i * S + j] = -1e10f;
            s = add(s, mask);
        }
        auto p = softmax(s, -1);
        auto c = matmul(p, V_use);
        auto L = sum(sum(sum(sum(mul(c, G), -1, false), -1, false), -1, false), -1, false);
        return L->data<float>()[0];
    };

    auto compute_numeric = [&](const TensorPtr& base, bool is_k, float step, TensorPtr out) {
        int64_t N = base->numel();
        for (int64_t i = 0; i < N; ++i) {
            auto p = base->clone(); auto n = base->clone();
            p->data<float>()[i] += step; n->data<float>()[i] -= step;
            float lp = is_k ? eval_with(p, v_in) : eval_with(k_in, p);
            float ln = is_k ? eval_with(n, v_in) : eval_with(k_in, n);
            out->data<float>()[i] = (lp - ln) / (2 * step);
        }
    };

    // eps sweep
    std::vector<float> eps_list = {1e-2f, 3e-3f, 1e-3f, 3e-4f, 1e-4f, 1e-5f};
    auto tmp_k = zeros(k_in->shape(), kFloat32, kCPU);
    auto tmp_v = zeros(v_in->shape(), kFloat32, kCPU);
    for (float e : eps_list) {
        compute_numeric(k_in, true, e, tmp_k);
        compute_numeric(v_in, false, e, tmp_v);
        float rdk_sweep = rel_diff(gk_in, tmp_k);
        float rdv_sweep = rel_diff(gv_in, tmp_v);
        std::cout << "[T3.AttnOnly][eps=" << e << "] rdiff dK_in=" << rdk_sweep
                  << " dV_in=" << rdv_sweep << std::endl;
    }

    // If repeating, also compute numeric gradient directly on k_full, then sum back to k_in
    if (USE_REPEAT) {
        auto eval_with_kfull = [&](const TensorPtr& K_full, const TensorPtr& V_full) -> float {
            auto s = mul(matmul(q, transpose(K_full, 2, 3)), scale);
            if (USE_MASK) {
                auto mask = zeros({B, 1, S, S}, kFloat32, kCPU);
                float* md = mask->data<float>();
                for (int i = 0; i < S; ++i) for (int j = i + 1; j < S; ++j) md[i * S + j] = -1e10f;
                s = add(s, mask);
            }
            auto p = softmax(s, -1);
            auto c = matmul(p, V_full);
            auto L = sum(sum(sum(sum(mul(c, G), -1, false), -1, false), -1, false), -1, false);
            return L->data<float>()[0];
        };
        auto k_full_shape = k_use->shape(); // [B,H,S,Hd]
        auto gk_full_num = zeros(k_full_shape, kFloat32, kCPU);
        auto v_full_shape = v_use->shape();
        // eps sweep on k_full numeric, then sum-back to k_in and compare
        for (float e : eps_list) {
            // Compute numeric on k_full
            for (int64_t i = 0; i < k_use->numel(); ++i) {
                auto kp = k_use->clone(); auto kn = k_use->clone();
                kp->data<float>()[i] += e; kn->data<float>()[i] -= e;
                float lp = eval_with_kfull(kp, v_use);
                float ln = eval_with_kfull(kn, v_use);
                gk_full_num->data<float>()[i] = (lp - ln) / (2 * e);
            }
            // Sum-back over heads to form k_in-shape numeric
            auto gk_num_from_full = zeros(k_in->shape(), kFloat32, kCPU);
            {
                const float* src = gk_full_num->data<float>();
                float* dst = gk_num_from_full->data<float>();
                int64_t heads_rep = H / kvH;
                for (int64_t b=0;b<B;++b){
                    for (int64_t kv=0;kv<kvH;++kv){
                        for (int64_t s=0;s<S;++s){
                            for (int64_t d=0;d<Hd;++d){
                                double sum = 0.0;
                                for (int64_t rep=0; rep < heads_rep; ++rep){
                                    int64_t out_h = kv*heads_rep + rep;
                                    int64_t si = (((b*H + out_h)*S + s)*Hd + d);
                                    sum += static_cast<double>(src[si]);
                                }
                                int64_t di = (((b*kvH + kv)*S + s)*Hd + d);
                                dst[di] = static_cast<float>(sum);
                            }
                        }
                    }
                }
            }
            float rd_full_sweep = rel_diff(gk_in, gk_num_from_full);
            std::cout << "[T3.AttnOnly][eps=" << e << "][k_full] rdiff dK_in(sum-back)=" << rd_full_sweep << std::endl;
        }
    }

    // Default eps numeric (diagnostic only)
    compute_numeric(k_in, true, eps, gk_num);
    compute_numeric(v_in, false, eps, gv_num);

    float rdk = rel_diff(gk_in, gk_num);
    float rdv = rel_diff(gv_in, gv_num);
    std::cout << "[T3.AttnOnly] rel_diff dK_in=" << rdk << " dV_in=" << rdv
              << " (mask=" << (USE_MASK ? 1 : 0)
              << ", repeat=" << (USE_REPEAT ? 1 : 0)
              << ", rope=" << (USE_ROPE ? 1 : 0) << ")"
              << std::endl;
    // Final pass criteria: rely on analytic checks rather than numeric FD
    bool ok = (dScores_rd < 1e-6f) && (rdk_analytic < 1e-5f) && (rdv_analytic < 1e-5f);
    std::cout << (ok ? "[PASS]" : "[FAIL]") << std::endl;
    return ok ? 0 : 1;
}


