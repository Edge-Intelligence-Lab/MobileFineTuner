#pragma once

#include "tensor.h"
#include <memory>
#include <vector>

namespace ops {

class Tensor;
using TensorPtr = std::shared_ptr<Tensor>;

class BackwardFunction {
public:
    virtual ~BackwardFunction() = default;
    virtual std::vector<TensorPtr> apply(const TensorPtr& grad_output) = 0;
};

using BackwardFunctionPtr = std::shared_ptr<BackwardFunction>;

class AddBackward : public BackwardFunction {
private:
    std::vector<int64_t> shape_a_, shape_b_;

public:
    AddBackward(const std::vector<int64_t>& shape_a, const std::vector<int64_t>& shape_b)
        : shape_a_(shape_a), shape_b_(shape_b) {}

    std::vector<TensorPtr> apply(const TensorPtr& grad_output) override;
};

class MulBackward : public BackwardFunction {
private:
    TensorPtr a_, b_;

public:
    MulBackward(const TensorPtr& a, const TensorPtr& b) : a_(a), b_(b) {}

    std::vector<TensorPtr> apply(const TensorPtr& grad_output) override;
};

class SubBackward : public BackwardFunction {
private:
    std::vector<int64_t> shape_a_, shape_b_;

public:
    SubBackward(const std::vector<int64_t>& shape_a, const std::vector<int64_t>& shape_b)
        : shape_a_(shape_a), shape_b_(shape_b) {}

    std::vector<TensorPtr> apply(const TensorPtr& grad_output) override;
};

class MatmulBackward : public BackwardFunction {
private:
    TensorPtr a_, b_;

public:
    MatmulBackward(const TensorPtr& a, const TensorPtr& b) : a_(a), b_(b) {}

    std::vector<TensorPtr> apply(const TensorPtr& grad_output) override;
};

class ReluBackward : public BackwardFunction {
private:
    TensorPtr input_;

public:
    ReluBackward(const TensorPtr& input) : input_(input) {}

    std::vector<TensorPtr> apply(const TensorPtr& grad_output) override;
};

class GeluBackward : public BackwardFunction {
private:
    TensorPtr input_;

public:
    GeluBackward(const TensorPtr& input) : input_(input) {}

    std::vector<TensorPtr> apply(const TensorPtr& grad_output) override;
};

class SigmoidBackward : public BackwardFunction {
private:
    TensorPtr output_;

public:
    SigmoidBackward(const TensorPtr& output) : output_(output) {}

    std::vector<TensorPtr> apply(const TensorPtr& grad_output) override;
};

class TanhBackward : public BackwardFunction {
private:
    TensorPtr output_;
public:
    explicit TanhBackward(const TensorPtr& output) : output_(output) {}
    std::vector<TensorPtr> apply(const TensorPtr& grad_output) override;
};

class SiLUBackward : public BackwardFunction {
private:
    TensorPtr input_;
public:
    explicit SiLUBackward(const TensorPtr& input) : input_(input) {}
    std::vector<TensorPtr> apply(const TensorPtr& grad_output) override;
};

class DivBackward : public BackwardFunction {
private:
    TensorPtr a_;
    TensorPtr b_;
public:
    DivBackward(const TensorPtr& a, const TensorPtr& b) : a_(a), b_(b) {}
    std::vector<TensorPtr> apply(const TensorPtr& grad_output) override;
};

class SqrtBackward : public BackwardFunction {
private:
    TensorPtr input_;
public:
    explicit SqrtBackward(const TensorPtr& input) : input_(input) {}
    std::vector<TensorPtr> apply(const TensorPtr& grad_output) override;
};

class ExpBackward : public BackwardFunction {
private:
    TensorPtr output_;
public:
    explicit ExpBackward(const TensorPtr& output) : output_(output) {}
    std::vector<TensorPtr> apply(const TensorPtr& grad_output) override;
};

class LogBackward : public BackwardFunction {
private:
    TensorPtr input_;
public:
    explicit LogBackward(const TensorPtr& input) : input_(input) {}
    std::vector<TensorPtr> apply(const TensorPtr& grad_output) override;
};

class PowBackward : public BackwardFunction {
private:
    TensorPtr input_;
    float exponent_;
public:
    PowBackward(const TensorPtr& input, float exponent) : input_(input), exponent_(exponent) {}
    std::vector<TensorPtr> apply(const TensorPtr& grad_output) override;
};

class AbsBackward : public BackwardFunction {
private:
    TensorPtr input_;
public:
    explicit AbsBackward(const TensorPtr& input) : input_(input) {}
    std::vector<TensorPtr> apply(const TensorPtr& grad_output) override;
};

class SwiGLUBackward : public BackwardFunction {
private:
    TensorPtr gate_;
    TensorPtr up_;
public:
    SwiGLUBackward(const TensorPtr& gate, const TensorPtr& up) : gate_(gate), up_(up) {}
    std::vector<TensorPtr> apply(const TensorPtr& grad_output) override;
};

// y = scalar / x
class ScalarOverTensorBackward : public BackwardFunction {
private:
    TensorPtr input_;
    float scalar_;
public:
    ScalarOverTensorBackward(const TensorPtr& input, float scalar) : input_(input), scalar_(scalar) {}
    std::vector<TensorPtr> apply(const TensorPtr& grad_output) override;
};

class SoftmaxBackward : public BackwardFunction {
private:
    TensorPtr output_;
    int dim_;

public:
    SoftmaxBackward(const TensorPtr& output, int dim) : output_(output), dim_(dim) {}

    std::vector<TensorPtr> apply(const TensorPtr& grad_output) override;
};

class LayerNormBackward : public BackwardFunction {
private:
    TensorPtr input_, weight_, bias_;
    TensorPtr mean_, var_;
    float eps_;

public:
    LayerNormBackward(const TensorPtr& input, const TensorPtr& weight,
                     const TensorPtr& bias, const TensorPtr& mean,
                     const TensorPtr& var, float eps)
        : input_(input), weight_(weight), bias_(bias), mean_(mean), var_(var), eps_(eps) {}

    std::vector<TensorPtr> apply(const TensorPtr& grad_output) override;
};

// Backward for RMSNorm: y = (x / sqrt(mean(x^2)+eps)) * weight
class RMSNormBackward : public BackwardFunction {
private:
    TensorPtr input_;
    TensorPtr weight_;
    float eps_;
public:
    RMSNormBackward(const TensorPtr& input, const TensorPtr& weight, float eps)
        : input_(input), weight_(weight), eps_(eps) {}
    std::vector<TensorPtr> apply(const TensorPtr& grad_output) override;
};

// Backward for RMSNorm with affine scale only: y = x_hat * w
class RMSNormAffineBackward : public BackwardFunction {
private:
    TensorPtr input_;
    TensorPtr weight_;
    float eps_;
public:
    RMSNormAffineBackward(const TensorPtr& input, const TensorPtr& weight, float eps)
        : input_(input), weight_(weight), eps_(eps) {}
    std::vector<TensorPtr> apply(const TensorPtr& grad_output) override;
};

class SliceBackward : public BackwardFunction {
private:
    TensorPtr input_;
    int dim_;
    int64_t start_;
    int64_t length_;
public:
    SliceBackward(const TensorPtr& input, int dim, int64_t start, int64_t length)
        : input_(input), dim_(dim), start_(start), length_(length) {}
    std::vector<TensorPtr> apply(const TensorPtr& grad_output) override;
};

class LinearBackward : public BackwardFunction {
private:
    TensorPtr input_, weight_, bias_;

public:
    LinearBackward(const TensorPtr& input, const TensorPtr& weight, const TensorPtr& bias)
        : input_(input), weight_(weight), bias_(bias) {}

    std::vector<TensorPtr> apply(const TensorPtr& grad_output) override;
};

class MSELossBackward : public BackwardFunction {
private:
    TensorPtr input_, target_;
    std::string reduction_;

public:
    MSELossBackward(const TensorPtr& input, const TensorPtr& target, const std::string& reduction)
        : input_(input), target_(target), reduction_(reduction) {}

    std::vector<TensorPtr> apply(const TensorPtr& grad_output) override;
};

class TransposeBackward : public BackwardFunction {
private:
    int dim0_, dim1_;

public:
    TransposeBackward(int dim0, int dim1) : dim0_(dim0), dim1_(dim1) {}

    std::vector<TensorPtr> apply(const TensorPtr& grad_output) override;
};

class ReshapeBackward : public BackwardFunction {
private:
    std::vector<int64_t> original_shape_;

public:
    ReshapeBackward(const std::vector<int64_t>& original_shape) : original_shape_(original_shape) {}

    std::vector<TensorPtr> apply(const TensorPtr& grad_output) override;
};

class CrossEntropyLossBackward : public BackwardFunction {
private:
    TensorPtr input_;
    TensorPtr target_;
    std::string reduction_;

public:
    CrossEntropyLossBackward(const TensorPtr& input, const TensorPtr& target, const std::string& reduction)
        : input_(input), target_(target), reduction_(reduction) {}

    std::vector<TensorPtr> apply(const TensorPtr& grad_output) override;
};

class LogSoftmaxBackward : public BackwardFunction {
private:
    TensorPtr input_;
    TensorPtr output_;
    int dim_;

public:
    LogSoftmaxBackward(const TensorPtr& input, const TensorPtr& output, int dim)
        : input_(input), output_(output), dim_(dim) {}

    std::vector<TensorPtr> apply(const TensorPtr& grad_output) override;
};

class PermuteBackward : public BackwardFunction {
private:
    std::vector<int> inv_dims_;
public:
    explicit PermuteBackward(const std::vector<int>& inv_dims) : inv_dims_(inv_dims) {}
    std::vector<TensorPtr> apply(const TensorPtr& grad_output) override;
};

class SumBackward : public BackwardFunction {
private:
    std::vector<int64_t> input_shape_;
    int dim_;
    bool keepdim_;

public:
    SumBackward(const std::vector<int64_t>& input_shape, int dim, bool keepdim)
        : input_shape_(input_shape), dim_(dim), keepdim_(keepdim) {}

    std::vector<TensorPtr> apply(const TensorPtr& grad_output) override;
};

class BatchNormBackward : public BackwardFunction {
private:
    TensorPtr input_;
    TensorPtr weight_;
    TensorPtr mean_;
    TensorPtr var_;
    float eps_;
public:
    BatchNormBackward(const TensorPtr& input, const TensorPtr& weight,
                      const TensorPtr& mean, const TensorPtr& var, float eps)
        : input_(input), weight_(weight), mean_(mean), var_(var), eps_(eps) {}
    std::vector<TensorPtr> apply(const TensorPtr& grad_output) override;
};

class NLLLossBackward : public BackwardFunction {
private:
    TensorPtr input_;
    TensorPtr target_;
    std::string reduction_;
public:
    NLLLossBackward(const TensorPtr& input, const TensorPtr& target, const std::string& reduction)
        : input_(input), target_(target), reduction_(reduction) {}
    std::vector<TensorPtr> apply(const TensorPtr& grad_output) override;
};

// New: scale backward for unary ops like mul(tensor, scalar) and div
class ScaleBackward : public BackwardFunction {
private:
    float scale_;
public:
    explicit ScaleBackward(float scale) : scale_(scale) {}
    std::vector<TensorPtr> apply(const TensorPtr& grad_output) override;
};

// New: pass-through backward for unary ops like add/sub with scalar
class PassThroughBackward : public BackwardFunction {
public:
    PassThroughBackward() = default;
    std::vector<TensorPtr> apply(const TensorPtr& grad_output) override;
};

class DropoutBackward : public BackwardFunction {
private:
    TensorPtr mask_;
    float scale_;
public:
    DropoutBackward(const TensorPtr& mask, float scale) : mask_(mask), scale_(scale) {}
    std::vector<TensorPtr> apply(const TensorPtr& grad_output) override;
};

// Backward for apply_mask: y = input + mask (mask is constant w.r.t. gradients)
class ApplyMaskBackward : public BackwardFunction {
private:
    TensorPtr input_;
public:
    explicit ApplyMaskBackward(const TensorPtr& input) : input_(input) {}
    std::vector<TensorPtr> apply(const TensorPtr& grad_output) override;
};

// New: matmul_rhs_T backward for y = a @ b^T (b is [N,K])
class MatmulRhsTBackward : public BackwardFunction {
private:
    TensorPtr a_;
    TensorPtr b_;
public:
    MatmulRhsTBackward(const TensorPtr& a, const TensorPtr& b) : a_(a), b_(b) {}
    std::vector<TensorPtr> apply(const TensorPtr& grad_output) override;
};

class LoRALinearBackward : public BackwardFunction {
private:
    TensorPtr input_;
    TensorPtr weight_;
    TensorPtr lora_A_;
    TensorPtr lora_B_;
    TensorPtr bias_;
    float alpha_;

public:
    LoRALinearBackward(const TensorPtr& input, const TensorPtr& weight,
                      const TensorPtr& lora_A, const TensorPtr& lora_B,
                      float alpha, const TensorPtr& bias)
        : input_(input), weight_(weight), lora_A_(lora_A), lora_B_(lora_B),
          bias_(bias), alpha_(alpha) {}

    std::vector<TensorPtr> apply(const TensorPtr& grad_output) override;
};

// Backward for repeat_kv_heads: expands KV heads by repeating; grad should sum back
class RepeatKVHeadsBackward : public BackwardFunction {
private:
    int repeat_factor_;
public:
    explicit RepeatKVHeadsBackward(int repeat_factor) : repeat_factor_(repeat_factor) {}
    std::vector<TensorPtr> apply(const TensorPtr& grad_output) override;
};

// Backward for RoPE: y = R(theta) * x, grad_x = R(theta)^T * grad_y
class ApplyRoPEBackward : public BackwardFunction {
private:
    int head_dim_;
    float rope_theta_;
public:
    ApplyRoPEBackward(int seq_len [[maybe_unused]], int head_dim, float rope_theta)
        : head_dim_(head_dim), rope_theta_(rope_theta) {}
    std::vector<TensorPtr> apply(const TensorPtr& grad_output) override;
};

class MemoryFirstMLPBackward : public BackwardFunction {
private:
    TensorPtr input_, fc_weight_, fc_bias_, proj_weight_, proj_bias_;
    int chunk_size_;
    int64_t batch_seq_, n_embd_, n_inner_;
public:
    MemoryFirstMLPBackward(const TensorPtr& input, const TensorPtr& fc_weight,
                          const TensorPtr& fc_bias, const TensorPtr& proj_weight,
                          const TensorPtr& proj_bias, int chunk_size,
                          int64_t batch_seq, int64_t n_embd, int64_t n_inner)
        : input_(input), fc_weight_(fc_weight), fc_bias_(fc_bias),
          proj_weight_(proj_weight), proj_bias_(proj_bias), chunk_size_(chunk_size),
          batch_seq_(batch_seq), n_embd_(n_embd), n_inner_(n_inner) {}
    std::vector<TensorPtr> apply(const TensorPtr& grad_output) override;
};

class GraphNode {
public:
    std::vector<std::shared_ptr<GraphNode>> inputs;
    BackwardFunctionPtr backward_fn;
    TensorPtr tensor;

    GraphNode(const TensorPtr& t) : tensor(t) {}

    void add_input(const std::shared_ptr<GraphNode>& input) {
        inputs.push_back(input);
    }

    void set_backward_fn(BackwardFunctionPtr fn) {
        backward_fn = fn;
    }
};

using GraphNodePtr = std::shared_ptr<GraphNode>;

GraphNodePtr create_graph_node(const TensorPtr& tensor);

void set_tensor_graph_info(const TensorPtr& result,
                          const std::vector<TensorPtr>& inputs,
                          BackwardFunctionPtr backward_fn);

void accumulate_gradient(const TensorPtr& tensor, const TensorPtr& grad);

TensorPtr sum_to_shape(const TensorPtr& tensor, const std::vector<int64_t>& target_shape);

}
