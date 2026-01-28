#include "../core/ops.h"
#include "../core/autograd_engine.h"
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <cassert>

using namespace ops;

static float rel_diff(const TensorPtr& a, const TensorPtr& b) {
    const float* pa = a->data<float>();
    const float* pb = b->data<float>();
    double na = 0.0, nb = 0.0, nd = 0.0;
    int64_t n = a->numel();
    for (int64_t i = 0; i < n; ++i) {
        double va = static_cast<double>(pa[i]);
        double vb = static_cast<double>(pb[i]);
        nd += (va - vb) * (va - vb);
        nb += vb * vb;
    }
    if (nb == 0.0) return std::sqrt(nd);
    return static_cast<float>(std::sqrt(nd / nb));
}

int main() {
    // Small-dim attention: B=1, H=1, S=3, Hd=4
    const int64_t B = 1, H = 1, S = 3, Hd = 4;
    const float scale = std::pow(256.0f, -0.5f); // Gemma3 default query_pre_attn_scalar=256

    // Prepare q, k with requires_grad
    auto q = zeros({B, H, S, Hd}, kFloat32, kCPU);
    auto k = zeros({B, H, S, Hd}, kFloat32, kCPU);
    q->set_requires_grad(true);
    k->set_requires_grad(true);

    // Fill deterministic values
    {
        float* qd = q->data<float>();
        float* kd = k->data<float>();
        for (int i = 0; i < q->numel(); ++i) qd[i] = 0.1f * (i % 7 - 3);
        for (int i = 0; i < k->numel(); ++i) kd[i] = 0.07f * (i % 5 - 2);
    }

    // scores = (q @ k^T) * scale
    auto k_t = transpose(k, 2, 3);                          // [B,H,Hd,S]
    auto scores_mat = matmul(q, k_t);                        // [B,H,S,S]
    scores_mat->retain_grad();
    auto scores = mul(scores_mat, scale);
    scores->retain_grad();

    // Softmax over last dim
    auto probs = softmax(scores, -1);                        // [B,H,S,S]

    // Upstream grad on probs (random deterministic)
    auto g_probs = zeros(probs->shape(), kFloat32, kCPU);
    {
        float* gp = g_probs->data<float>();
        for (int i = 0; i < g_probs->numel(); ++i) gp[i] = 0.03f * ((i % 11) - 5);
    }

    // Run backward through engine: dL/dq, dL/dk
    {
        using namespace ops::autograd;
        Engine::instance().run_backward({probs}, {g_probs});
    }

    auto gq_engine = q->grad();
    auto gk_engine = k->grad();
    auto gscores_engine = scores->grad();      // dL/dscores
    auto gmat_engine = scores_mat->grad();     // dL/d(scores_mat)
    if (!gscores_engine) gscores_engine = zeros(scores->shape(), kFloat32, kCPU);
    if (!gmat_engine) gmat_engine = zeros(scores_mat->shape(), kFloat32, kCPU);

    // Reference grads (analytic):
    // grad_scores = softmax_backward(g_probs, probs)
    auto grad_scores = zeros(scores->shape(), kFloat32, kCPU);
    {
        // y * (go - sum(go*y))
        auto go_y = mul(g_probs, probs);
        int last_dim = static_cast<int>(go_y->shape().size()) - 1;
        auto sum_go = sum(go_y, last_dim, true);
        auto diff = sub(g_probs, sum_go);
        auto tmp = mul(probs, diff);
        // tmp is grad wrt scores
        // account for scale: dL/dM = scale * dL/dscores
        grad_scores = tmp;
    }

    auto grad_mat = mul(grad_scores, scale);   // dL/d(scores_mat)

    // gq_ref = grad_scores @ k
    // gk_ref = grad_scores^T @ q
    auto k_ref = k; // [B,H,S,Hd]
    auto grad_q_ref = matmul(grad_mat, k_ref);                   // [B,H,S,Hd]
    auto grad_k_ref = matmul(transpose(grad_mat, -2, -1), q);    // [B,H,S,Hd]

    float rds = rel_diff(gscores_engine, grad_scores);
    float rdm = rel_diff(gmat_engine, grad_mat);
    float rdq = rel_diff(gq_engine, grad_q_ref);
    float rdk = rel_diff(gk_engine, grad_k_ref);

    std::cout << "[T5.QKT] rel_diff dScores=" << rds
              << " dMat=" << rdm
              << " dq=" << rdq
              << " dk=" << rdk << std::endl;

    // Tight threshold
    bool ok = (rdq < 1e-6f) && (rdk < 1e-6f);
    std::cout << (ok ? "[PASS]" : "[FAIL]") << std::endl;

    // Additional: standalone softmax backward check
    auto x = zeros({2, 3}, kFloat32, kCPU);
    x->set_requires_grad(true);
    {
        float* xd = x->data<float>();
        for (int i = 0; i < x->numel(); ++i) xd[i] = 0.1f * (i - 2);
    }
    auto y = softmax(x, -1);
    auto gy = zeros(y->shape(), kFloat32, kCPU);
    {
        float* gd = gy->data<float>();
        for (int i = 0; i < gy->numel(); ++i) gd[i] = 0.05f * ((i % 5) - 2);
    }
    {
        using namespace ops::autograd;
        Engine::instance().run_backward({y}, {gy});
    }
    auto gx_engine = x->grad();
    // analytic
    auto go_y2 = mul(gy, y);
    int last_dim2 = static_cast<int>(go_y2->shape().size()) - 1;
    auto sum_go2 = sum(go_y2, last_dim2, true);
    auto diff2 = sub(gy, sum_go2);
    auto gx_ref = mul(y, diff2);
    float rdsf = rel_diff(gx_engine, gx_ref);
    std::cout << "[SoftmaxOnly] rel_diff dX=" << rdsf << std::endl;

    // Numeric check
    auto numeric = zeros(x->shape(), kFloat32, kCPU);
    {
        const float eps = 1e-4f;
        for (int64_t i = 0; i < x->numel(); ++i) {
            auto x_pos = x->clone(); auto x_neg = x->clone();
            x_pos->data<float>()[i] += eps;
            x_neg->data<float>()[i] -= eps;
            auto y_pos = softmax(x_pos, -1);
            auto y_neg = softmax(x_neg, -1);
            // loss = sum(y * gy)
            auto lp = sum(mul(y_pos, gy));
            auto ln = sum(mul(y_neg, gy));
            float lpos = lp->data<float>()[0];
            float lneg = ln->data<float>()[0];
            numeric->data<float>()[i] = (lpos - lneg) / (2 * eps);
        }
    }
    float rdn = rel_diff(gx_engine, numeric);
    std::cout << "[SoftmaxOnly] rel_diff dX vs numeric=" << rdn << std::endl;

    // Numeric check for full chain wrt q and k (small dims)
    // Define scalar L = sum(probs * G) with fixed G
    auto G = zeros(probs->shape(), kFloat32, kCPU);
    {
        float* gd = G->data<float>();
        for (int i = 0; i < G->numel(); ++i) gd[i] = 0.02f * ((i % 7) - 3);
    }
    // Clear previous grads and re-run backward with G to compare apples-to-apples with numeric
    {
        using namespace ops::autograd;
        if (q->grad()) q->zero_grad();
        if (k->grad()) k->zero_grad();
        // Rebuild forward chain to avoid stale graph state
        auto k_t2 = transpose(k, 2, 3);
        auto scores_mat2 = matmul(q, k_t2);
        auto scores2 = mul(scores_mat2, scale);
        auto probs2 = softmax(scores2, -1);
        Engine::instance().run_backward({probs2}, {G});
    }
    gq_engine = q->grad();
    gk_engine = k->grad();
    auto numeric_q = zeros(q->shape(), kFloat32, kCPU);
    auto numeric_k = zeros(k->shape(), kFloat32, kCPU);
    const float eps = 1e-5f;
    // Numeric wrt q
    for (int64_t i = 0; i < q->numel(); ++i) {
        auto qp = q->clone(); auto qn = q->clone();
        qp->data<float>()[i] += eps;
        qn->data<float>()[i] -= eps;
        auto sp = mul(matmul(qp, transpose(k, 2, 3)), scale);
        auto sn = mul(matmul(qn, transpose(k, 2, 3)), scale);
        auto yp = softmax(sp, -1);
        auto yn = softmax(sn, -1);
        auto lp = sum(mul(yp, G));
        auto ln = sum(mul(yn, G));
        float lpos = lp->data<float>()[0];
        float lneg = ln->data<float>()[0];
        numeric_q->data<float>()[i] = (lpos - lneg) / (2 * eps);
    }
    // Numeric wrt k
    for (int64_t i = 0; i < k->numel(); ++i) {
        auto kp = k->clone(); auto kn = k->clone();
        kp->data<float>()[i] += eps;
        kn->data<float>()[i] -= eps;
        auto sp = mul(matmul(q, transpose(kp, 2, 3)), scale);
        auto sn = mul(matmul(q, transpose(kn, 2, 3)), scale);
        auto yp = softmax(sp, -1);
        auto yn = softmax(sn, -1);
        auto lp = sum(mul(yp, G));
        auto ln = sum(mul(yn, G));
        float lpos = lp->data<float>()[0];
        float lneg = ln->data<float>()[0];
        numeric_k->data<float>()[i] = (lpos - lneg) / (2 * eps);
    }
    float rdqn = rel_diff(gq_engine, numeric_q);
    float rdkn = rel_diff(gk_engine, numeric_k);
    std::cout << "[T5.QKT] rel_diff dq vs numeric=" << rdqn
              << " dk vs numeric=" << rdkn << std::endl;

    // Isolate matmul backward (no softmax): C = A @ B, L = sum(C * M)
    auto A = zeros(q->shape(), kFloat32, kCPU); A->set_requires_grad(true);
    auto Bten = zeros(k_t->shape(), kFloat32, kCPU); Bten->set_requires_grad(true);
    {
        float* ad = A->data<float>(); float* bd = Bten->data<float>();
        for (int i = 0; i < A->numel(); ++i) ad[i] = 0.1f * (i % 9 - 4);
        for (int i = 0; i < Bten->numel(); ++i) bd[i] = 0.07f * (i % 7 - 3);
    }
    auto C = matmul(A, Bten);
    auto M = zeros(C->shape(), kFloat32, kCPU);
    {
        float* md = M->data<float>();
        for (int i = 0; i < M->numel(); ++i) md[i] = 0.03f * (i % 5 - 2);
    }
    // Backprop with provided grad_output M
    {
        using namespace ops::autograd;
        Engine::instance().run_backward({C}, {M});
    }
    auto gA_engine = A->grad();
    auto gB_engine = Bten->grad();
    // Numeric finite differences for A
    auto gA_num = zeros(A->shape(), kFloat32, kCPU);
    for (int64_t i = 0; i < A->numel(); ++i) {
        auto Ap = A->clone(); auto An = A->clone();
        Ap->data<float>()[i] += eps;
        An->data<float>()[i] -= eps;
        auto Cp = matmul(Ap, Bten);
        auto Cn = matmul(An, Bten);
        auto lp = sum(mul(Cp, M));
        auto ln = sum(mul(Cn, M));
        float lpos = lp->data<float>()[0];
        float lneg = ln->data<float>()[0];
        gA_num->data<float>()[i] = (lpos - lneg) / (2 * eps);
    }
    // Numeric finite differences for B
    auto gB_num = zeros(Bten->shape(), kFloat32, kCPU);
    for (int64_t i = 0; i < Bten->numel(); ++i) {
        auto Bp = Bten->clone(); auto Bn = Bten->clone();
        Bp->data<float>()[i] += eps;
        Bn->data<float>()[i] -= eps;
        auto Cp = matmul(A, Bp);
        auto Cn = matmul(A, Bn);
        auto lp = sum(mul(Cp, M));
        auto ln = sum(mul(Cn, M));
        float lpos = lp->data<float>()[0];
        float lneg = ln->data<float>()[0];
        gB_num->data<float>()[i] = (lpos - lneg) / (2 * eps);
    }
    float rdAm = rel_diff(gA_engine, gA_num);
    float rdBm = rel_diff(gB_engine, gB_num);
    std::cout << "[MatmulOnly] rel_diff dA vs numeric=" << rdAm
              << " dB vs numeric=" << rdBm << std::endl;

    return (rdsf < 1e-6f && rdn < 1e-4f && rdqn < 1e-3f && rdkn < 1e-3f && rdAm < 1e-6f && rdBm < 1e-6f) ? 0 : 1;
}


