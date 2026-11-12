#include "memory_efficient_attention.h"
#include "ops.h"
#include "backward_functions.h"
#ifdef USE_NEW_AUTOGRAD_ENGINE
#include "autograd_engine.h"
#endif
#include <algorithm>
#include <cmath>
#include <limits>

namespace ops {
namespace {
int64_t offset(const TensorPtr& t, int64_t b, int64_t h, int64_t s, int64_t d) {
    const auto& shape = t->shape();
    return ((b * shape[1] + h) * shape[2] + s) * shape[3] + d;
}
void check_tensor(const TensorPtr& t) {
    if (!t || t->dtype() != kFloat32 || !t->device().is_cpu())
        throw std::invalid_argument("streaming attention requires CPU FP32 tensors");
}

// Forward and backward share masking and normalization; only O(S) scratch is used.
void probabilities(const TensorPtr& q, const TensorPtr& k, const TensorPtr& mask,
                   const TensorPtr& pad, float scale, bool causal,
                   int64_t b, int64_t h, int64_t i, std::vector<float>& p) {
    const int64_t S = q->shape()[2], D = q->shape()[3];
    float maximum = -std::numeric_limits<float>::infinity();
    for (int64_t j = 0; j < S; ++j) {
        float score = 0;
        for (int64_t d = 0; d < D; ++d)
            score += q->data<float>()[offset(q,b,h,i,d)] * k->data<float>()[offset(k,b,h,j,d)];
        score *= scale;
        if (causal && j > i) score = -std::numeric_limits<float>::infinity();
        if (mask) score += mask->data<float>()[i * S + j];
        if (pad) score += pad->data<float>()[b * S + j];
        if (std::isnan(score) || score == std::numeric_limits<float>::infinity())
            throw std::runtime_error("streaming attention received non-finite logits");
        p[j] = score;
        maximum = std::max(maximum, score);
    }
    // An all -infinity row has no valid keys: output and gradient are zero.
    if (maximum == -std::numeric_limits<float>::infinity()) {
        std::fill(p.begin(), p.end(), 0.0f);
        return;
    }
    float sum = 0;
    for (auto& value : p) { value = std::exp(value - maximum); sum += value; }
    for (auto& value : p) value /= sum;
}

class StreamingBackward : public BackwardFunction {
public:
    StreamingBackward(TensorPtr q, TensorPtr k, TensorPtr v, TensorPtr mask,
                      TensorPtr pad, float scale, bool causal)
        : q_(q), k_(k), v_(v), mask_(mask), pad_(pad), scale_(scale), causal_(causal) {}
    std::vector<TensorPtr> apply(const TensorPtr& go) override {
        check_tensor(go);
        if (go->shape() != q_->shape()) throw std::invalid_argument("streaming attention gradient shape mismatch");
        auto dq = zeros(q_->shape()), dk = zeros(k_->shape()), dv = zeros(v_->shape());
        const auto shape = q_->shape();
        const int64_t S = shape[2], D = shape[3];
        std::vector<float> p(S), dp(S);
        for (int64_t b=0; b<shape[0]; ++b) for (int64_t h=0; h<shape[1]; ++h)
            for (int64_t i=0; i<S; ++i) {
                probabilities(q_,k_,mask_,pad_,scale_,causal_,b,h,i,p);
                double dot = 0;
                for (int64_t j=0; j<S; ++j) {
                    dp[j] = 0;
                    for (int64_t d=0; d<D; ++d) {
                        const float g = go->data<float>()[offset(go,b,h,i,d)];
                        dp[j] += g * v_->data<float>()[offset(v_,b,h,j,d)];
                        dv->data<float>()[offset(dv,b,h,j,d)] += p[j]*g;
                    }
                    dot += static_cast<double>(p[j])*static_cast<double>(dp[j]);
                }
                for (int64_t j=0; j<S; ++j) {
                    const float ds = (p[j]*(dp[j]-static_cast<float>(dot)))*scale_;
                    for (int64_t d=0; d<D; ++d) {
                        dq->data<float>()[offset(dq,b,h,i,d)] += ds*k_->data<float>()[offset(k_,b,h,j,d)];
                        dk->data<float>()[offset(dk,b,h,j,d)] += ds*q_->data<float>()[offset(q_,b,h,i,d)];
                    }
                }
            }
        return {dq,dk,dv};
    }
private:
    TensorPtr q_,k_,v_,mask_,pad_;
    float scale_;
    bool causal_;
};
} // namespace

void online_softmax_weighted_sum(const float* logits, const float* values,
        int64_t S, int64_t D, float* output, float maximum) {
    double sum = 0;
    for (int64_t j=0; j<S; ++j) sum += std::exp(logits[j]-maximum);
    for (int64_t j=0; j<S; ++j) for (int64_t d=0; d<D; ++d)
        output[d] += std::exp(logits[j]-maximum)/static_cast<float>(sum)*values[j*D+d];
}

TensorPtr memory_efficient_attention(const TensorPtr& q, const TensorPtr& k,
        const TensorPtr& v, const TensorPtr& mask, const MemoryEfficientAttentionConfig& config) {
    return memory_efficient_attention(q,k,v,mask,nullptr,config);
}

TensorPtr memory_efficient_attention(const TensorPtr& q, const TensorPtr& k,
        const TensorPtr& v, const TensorPtr& mask, const TensorPtr& pad,
        const MemoryEfficientAttentionConfig& config) {
    check_tensor(q); check_tensor(k); check_tensor(v);
    const auto shape = q->shape();
    if (shape.size()!=4 || k->shape()!=shape || v->shape()!=shape ||
        std::any_of(shape.begin(),shape.end(),[](int64_t d){return d<=0;}))
        throw std::invalid_argument("streaming attention expects matching positive [B,H,S,D] shapes");
    const int64_t B=shape[0], H=shape[1], S=shape[2], D=shape[3];
    if (mask) {
        check_tensor(mask);
        if (mask->shape()!=std::vector<int64_t>{S,S} || mask->requires_grad())
            throw std::invalid_argument("streaming attention mask must be a constant additive [S,S] tensor");
    }
    if (pad) {
        check_tensor(pad);
        if ((pad->shape()!=std::vector<int64_t>{B,S} && pad->shape()!=std::vector<int64_t>{B,1,1,S}) || pad->requires_grad())
            throw std::invalid_argument("streaming attention padding must be constant additive [B,S] or [B,1,1,S]");
    }
    if (!std::isfinite(config.scale)) throw std::invalid_argument("streaming attention scale must be finite");
    const float scale = config.scale>0 ? config.scale : 1/std::sqrt(static_cast<float>(D));
    // As in the previous training path, an explicit mask replaces automatic causality.
    const bool causal = !mask && config.use_causal_mask;
#ifndef USE_NEW_AUTOGRAD_ENGINE
    if (q->requires_grad() || k->requires_grad() || v->requires_grad()) {
        auto scores=mul(matmul(q,transpose(k,2,3)),scale);
        if (mask) scores=add(scores,mask);
        else if (causal) scores=add(scores,create_causal_mask(static_cast<int>(S),kFloat32,kCPU));
        if (pad) scores=add(scores,reshape(pad,{B,1,1,S}));
        return matmul(softmax(scores,-1),v);
    }
#endif
    auto out=zeros(shape);
    std::vector<float> p(S);
    for (int64_t b=0; b<B; ++b) for (int64_t h=0; h<H; ++h) for (int64_t i=0; i<S; ++i) {
        probabilities(q,k,mask,pad,scale,causal,b,h,i,p);
        for (int64_t j=0; j<S; ++j) for (int64_t d=0; d<D; ++d)
            out->data<float>()[offset(out,b,h,i,d)] += p[j]*v->data<float>()[offset(v,b,h,j,d)];
    }
#ifdef USE_NEW_AUTOGRAD_ENGINE
    if (q->requires_grad() || k->requires_grad() || v->requires_grad()) {
        out->set_requires_grad(true);
        autograd::Engine::instance().register_node(out,{q,k,v},
            std::make_shared<StreamingBackward>(q,k,v,mask,pad,scale,causal));
    }
#endif
    return out;
}
} // namespace ops
