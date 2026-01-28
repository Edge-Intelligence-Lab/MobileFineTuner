#include "../core/ops.h"
#include "../core/autograd_engine.h"
#include <cmath>
#include <iostream>
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

int main() {
    // Minimal settings: B=1, kv_heads=1 → repeat to H=4, S=3, Hd=4
    const int64_t B = 1, kvH = 1, H = 4, S = 3, Hd = 4;
    const float scale = std::pow(256.0f, -0.5f);

    auto q = zeros({B, H, S, Hd}, kFloat32, kCPU);      // will be requires_grad
    auto k_in = zeros({B, kvH, S, Hd}, kFloat32, kCPU); // input to repeat_kv (requires_grad)
    auto v_in = zeros({B, kvH, S, Hd}, kFloat32, kCPU);
    q->set_requires_grad(true);
    k_in->set_requires_grad(true);
    v_in->set_requires_grad(true);

    // Fill deterministic values
    {
        float* qd = q->data<float>();
        float* kd = k_in->data<float>();
        float* vd = v_in->data<float>();
        for (int i = 0; i < q->numel(); ++i) qd[i] = 0.05f * (i % 9 - 4);
        for (int i = 0; i < k_in->numel(); ++i) kd[i] = 0.04f * (i % 7 - 3);
        for (int i = 0; i < v_in->numel(); ++i) vd[i] = 0.03f * (i % 5 - 2);
    }

    auto k_full = repeat_kv_heads(k_in, H / kvH);   // [B,H,S,Hd]
    auto v_full = repeat_kv_heads(v_in, H / kvH);   // [B,H,S,Hd]
    k_full->retain_grad();
    v_full->retain_grad();
    auto k_t = transpose(k_full, 2, 3);             // [B,H,Hd,S]
    auto scores = mul(matmul(q, k_t), scale);       // [B,H,S,S]
    scores->retain_grad();
    // Simple causal mask (upper-triangular, j>i = -1e10)
    auto mask = zeros({B, 1, S, S}, kFloat32, kCPU);
    {
        float* md = mask->data<float>();
        for (int i = 0; i < S; ++i) {
            for (int j = i + 1; j < S; ++j) {
                md[i * S + j] = -1e10f;
            }
        }
    }
    // Disable mask for isolation
    // scores = add(scores, mask);
    auto probs = softmax(scores, -1);
    // Upstream gradient on context
    auto G = zeros({B, H, S, Hd}, kFloat32, kCPU);
    {
        float* gd = G->data<float>();
        for (int i = 0; i < G->numel(); ++i) gd[i] = 0.02f * (i % 11 - 5);
    }
    auto context = matmul(probs, v_full); // [B,H,S,Hd]
    auto loss = sum(mul(context, G), -1, false);
    loss = sum(loss, -1, false);
    loss = sum(loss, -1, false);
    loss = sum(loss, -1, false); // scalar

    {
        using namespace ops::autograd;
        Engine::instance().run_backward({loss}, {nullptr});
    }

    auto gq = q->grad();
    auto gk_in = k_in->grad();
    auto gv_in = v_in->grad();
    auto gk_full = k_full->grad();
    auto gv_full = v_full->grad();

    // Check that grad(k_in) equals sum over head repeats of grad(k_full)
    auto gk_from_full = zeros(k_in->shape(), kFloat32, kCPU);
    {
        const float* src = gk_full ? gk_full->data<float>() : nullptr;
        float* dst = gk_from_full->data<float>();
        int64_t heads_rep = H;
        for (int64_t b=0;b<B;++b){
            for (int64_t kv=0;kv<kvH;++kv){
                for (int64_t s=0;s<S;++s){
                    for (int64_t d=0;d<Hd;++d){
                        double sum = 0.0;
                        for (int64_t rep=0; rep < heads_rep; ++rep){
                            int64_t out_h = kv*heads_rep + rep;
                            int64_t si = (((b*heads_rep + out_h)*S + s)*Hd + d);
                            sum += src ? static_cast<double>(src[si]) : 0.0;
                        }
                        int64_t di = (((b*kvH + kv)*S + s)*Hd + d);
                        dst[di] = static_cast<float>(sum);
                    }
                }
            }
        }
    }
    float rdc = rel_diff(gk_in, gk_from_full);
    std::cout << "[RepeatKV] consistency grad K_in vs sum(K_full)=" << rdc << std::endl;

    // Analytic reference for grad wrt k_full using softmax backward:
    // dProbs = dContext @ V^T
    // dScores = probs * (dProbs - sum(dProbs*probs, -1, keepdim=True))
    // dMat = dScores * scale   // scores = (q @ k^T) * scale
    // gk_full_ref = (dMat^T) @ q
    auto v_t = transpose(v_full, -2, -1);                         // [B,H,Hd,S]
    auto dProbs = matmul(G, v_t);                                  // [B,H,S,S]
    auto prod = mul(dProbs, probs);
    auto sum_prod = sum(prod, -1, true);
    auto dScores = mul(probs, sub(dProbs, sum_prod));              // [B,H,S,S]
    // Check engine dScores vs analytic
    auto dScores_eng = scores->grad() ? scores->grad() : zeros(scores->shape(), kFloat32, kCPU);
    float rdd = rel_diff(dScores_eng, dScores);
    std::cout << "[RepeatKV] dScores engine vs ref=" << rdd << std::endl;
    // Explicit per-row softmax backward (no ops/broadcast) for verification
    auto dScores_loop = zeros(scores->shape(), kFloat32, kCPU);
    {
        const auto& shp = scores->shape(); // [B,H,S,S]
        int64_t batch = shp[0], heads = shp[1], rows = shp[2], cols = shp[3];
        const float* Y = probs->data<float>();
        const float* GO = dProbs->data<float>();
        float* DS = dScores_loop->data<float>();
        for (int64_t b=0;b<batch;++b){
            for (int64_t h=0;h<heads;++h){
                for (int64_t r=0;r<rows;++r){
                    const float* y = Y + (((b*heads+h)*rows + r) * cols);
                    const float* go = GO + (((b*heads+h)*rows + r) * cols);
                    float* ds = DS + (((b*heads+h)*rows + r) * cols);
                    double s = 0.0;
                    for (int64_t c=0;c<cols;++c) s += static_cast<double>(go[c]) * static_cast<double>(y[c]);
                    for (int64_t c=0;c<cols;++c) ds[c] = y[c] * (go[c] - static_cast<float>(s));
                }
            }
        }
    }
    float rdd2 = rel_diff(dScores_eng, dScores_loop);
    std::cout << "[RepeatKV] dScores engine vs loop=" << rdd2 << std::endl;
    auto dMat = mul(dScores, scale);
    auto dMat_T = transpose(dMat, -2, -1);                         // [B,H,S,S]
    auto gk_full_ref = matmul(dMat_T, q);                          // [B,H,S,Hd]
    float rdk_full = rel_diff(gk_full, gk_full_ref);
    std::cout << "[RepeatKV] analytic check grad K_full vs ref=" << rdk_full << std::endl;

    // Check matmul backward consistency using engine dScores directly
    auto gk_full_ref2 = matmul(transpose(mul(dScores_eng, scale), -2, -1), q);
    float rdk_full2 = rel_diff(gk_full, gk_full_ref2);
    std::cout << "[RepeatKV] engine gK_full vs (dScores_eng^T @ Q)=" << rdk_full2 << std::endl;
    // Numeric gradient check for k_in (repeat kv heads path)
    auto gk_num = zeros(k_in->shape(), kFloat32, kCPU);
    auto gv_num = zeros(v_in->shape(), kFloat32, kCPU);
    const float eps = 1e-5f;

    for (int64_t i = 0; i < k_in->numel(); ++i) {
        auto kp = k_in->clone(); auto kn = k_in->clone();
        kp->data<float>()[i] += eps;
        kn->data<float>()[i] -= eps;
        auto kfp = repeat_kv_heads(kp, H / kvH);
        auto kfn = repeat_kv_heads(kn, H / kvH);
        auto sp = mul(matmul(q, transpose(kfp, 2, 3)), scale);
        auto sn = mul(matmul(q, transpose(kfn, 2, 3)), scale);
        // no mask in isolation
        auto pp = softmax(sp, -1);
        auto pn = softmax(sn, -1);
        auto cp = matmul(pp, v_full);
        auto cn = matmul(pn, v_full);
        auto lp = sum(sum(sum(sum(mul(cp, G), -1, false), -1, false), -1, false), -1, false);
        auto ln = sum(sum(sum(sum(mul(cn, G), -1, false), -1, false), -1, false), -1, false);
        float lpos = lp->data<float>()[0];
        float lneg = ln->data<float>()[0];
        gk_num->data<float>()[i] = (lpos - lneg) / (2 * eps);
    }

    for (int64_t i = 0; i < v_in->numel(); ++i) {
        auto vp = v_in->clone(); auto vn = v_in->clone();
        vp->data<float>()[i] += eps;
        vn->data<float>()[i] -= eps;
        auto vfp = repeat_kv_heads(vp, H / kvH);
        auto vfn = repeat_kv_heads(vn, H / kvH);
        auto s = mul(matmul(q, transpose(k_full, 2, 3)), scale);
        auto p = softmax(s, -1);
        auto cp = matmul(p, vfp);
        auto cn = matmul(p, vfn);
        auto lp = sum(sum(sum(sum(mul(cp, G), -1, false), -1, false), -1, false), -1, false);
        auto ln = sum(sum(sum(sum(mul(cn, G), -1, false), -1, false), -1, false), -1, false);
        float lpos = lp->data<float>()[0];
        float lneg = ln->data<float>()[0];
        gv_num->data<float>()[i] = (lpos - lneg) / (2 * eps);
    }

    float rdk = rel_diff(gk_in, gk_num);
    float rdv = rel_diff(gv_in, gv_num);
    std::cout << "[RepeatKV+Softmax] rel_diff dK_in=" << rdk << " dV_in=" << rdv << std::endl;
    bool ok = (rdk < 1e-3f && rdv < 1e-3f);
    std::cout << (ok ? "[PASS]" : "[FAIL]") << std::endl;
    return ok ? 0 : 1;
}


