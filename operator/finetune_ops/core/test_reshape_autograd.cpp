#include "finetune_ops/core/ops.h"
#include "finetune_ops/nn/lora_linear.h"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <iostream>
#include <string>
#include <vector>

using namespace ops;
static int failures = 0;
static TensorPtr leaf() {
    auto x = ones({2, 3});
    x->set_requires_grad(true);
    return x;
}
static void check(const std::string& name, const TensorPtr& x, float expected) {
    double error = 0;
    const auto grad = x->grad();
    const bool layout_valid = grad && grad->shape() == x->shape() && grad->dtype() == x->dtype();
    if (layout_valid) for (int64_t i = 0; i < grad->numel(); ++i)
        error = std::isfinite(grad->data<float>()[i])
            ? std::max(error, double(std::abs(grad->data<float>()[i] - expected)))
            : std::numeric_limits<double>::infinity();
    const bool pass = layout_valid && error < 1e-6;
    failures += !pass;
    std::cout << "{\"test\":\"" << name << "\",\"expected\":" << expected
              << ",\"actual\":" << (layout_valid ? grad->data<float>()[0] : -999)
              << ",\"max_error\":" << error << ",\"passed\":" << (pass ? "true" : "false") << "}\n";
}
static void run_ops() {
    { auto x=leaf(); auto y=mul(x,1.0f); y->backward(ones(y->shape())); check("identity_op_control",x,1); }
    { auto x=leaf(); auto y=x->reshape({3,2}); y->backward(ones(y->shape())); check("member_reshape",x,1); }
    { auto x=leaf(); auto y=x->view({3,2}); y->backward(ones(y->shape())); check("member_view",x,1); }
    for (int depth=2; depth<=3; ++depth) {
        auto x=leaf(), y=x;
        for(int i=0;i<depth;++i) y=y->reshape(i%2 ? std::vector<int64_t>{2,3}:std::vector<int64_t>{3,2});
        y->backward(ones(y->shape())); check("member_reshape_depth_"+std::to_string(depth),x,1);
    }
    { auto x=leaf(); auto y=reshape(x->reshape({3,2}),{2,3}); y->backward(ones(y->shape())); check("mixed_member_ops_reshape",x,1); }
    for (int depth=1; depth<=3; ++depth) {
        auto x=leaf(), y=x;
        for(int i=0;i<depth;++i) y=reshape(y,i%2 ? std::vector<int64_t>{2,3}:std::vector<int64_t>{3,2});
        y->backward(ones(y->shape())); check("ops_reshape_depth_"+std::to_string(depth),x,1);
    }
    { auto x=leaf(); auto y=add(x,reshape(reshape(x,{3,2}),{2,3})); y->backward(ones(y->shape())); check("residual_two_reshapes",x,2); }
    { auto x=leaf(); auto y=add(x,x); y->backward(ones(y->shape())); check("legitimate_repeated_input_add",x,2); }
    { auto x=leaf(); auto y=mul(x,x); y->backward(ones(y->shape())); check("legitimate_repeated_input_mul",x,2); }
    { auto x=leaf(); auto y=transpose(x,0,1); y->backward(ones(y->shape())); check("transpose_control",x,1); }
    { auto x=leaf(); auto y=reshape(transpose(x,0,1),{6}); y->backward(ones(y->shape())); check("transpose_then_reshape",x,1); }
    { auto x=ones({2,3}); auto y=reshape(x,{3,2}); bool pass=!y->requires_grad(); failures+=!pass;
      std::cout << "{\"test\":\"frozen_input\",\"passed\":" << (pass?"true":"false") << "}\n"; }
}
static void run_lora(bool external_reshape) {
    auto W=ones({3,2}); auto A=ones({3,2}); auto B=ones({2,2}); auto x=ones({1,2,3});
    LoRALinear layer(W);
    layer.attach_lora(A,B,0.5f);
    auto y=layer.forward(x);
    if(external_reshape) y=reshape(y,{2,2});
    y->backward(ones(y->shape()));
    // d sum(0.5 * X A B)/dA = 0.5 * X^T ones B^T = 2;
    // d/dB = 0.5 * (X A)^T ones = 3. Independent analytic reference.
    check(external_reshape ? "lora_plus_reshape_A" : "local_lora_A_control",A,2);
    check(external_reshape ? "lora_plus_reshape_B" : "local_lora_B_control",B,3);
}
int main() {
    std::cout << std::setprecision(10);
    run_ops(); run_lora(false); run_lora(true);
    std::cout << "{\"failures\":" << failures << "}\n";
    return failures ? 1 : 0;
}
