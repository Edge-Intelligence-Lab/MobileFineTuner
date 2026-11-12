#include "finetune_ops/graph/qwen_model.h"
#include "finetune_ops/core/ops.h"
#include "finetune_ops/core/autograd_engine.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ops;
static bool gradient_agrees(double grad, double fd) {
    return std::isfinite(grad) && std::isfinite(fd)
        && std::abs(grad-fd)<=0.0003+0.02*std::abs(fd);
}
int main(int argc, char** argv) {
    if(argc!=2) throw std::runtime_error("Usage: qwen_fd output.csv");
    const double inf=std::numeric_limits<double>::infinity();
    const double nan=std::numeric_limits<double>::quiet_NaN();
    if(gradient_agrees(0.0,inf) || gradient_agrees(0.0,-inf)
       || gradient_agrees(0.0,nan) || gradient_agrees(inf,inf)
       || gradient_agrees(nan,0.0) || !gradient_agrees(0.0,0.0))
        throw std::runtime_error("Finite-difference comparison guard failed");
    QwenConfig cfg;
    cfg.vocab_size=32; cfg.hidden_size=16; cfg.intermediate_size=32;
    cfg.num_hidden_layers=2; cfg.num_attention_heads=4; cfg.num_key_value_heads=2;
    cfg.max_position_embeddings=32;
    QwenModel model(cfg);
    std::mt19937 rng(731);
    std::normal_distribution<float> random(0.0f, 0.1f);
    auto assign=[&](const std::string& key, std::vector<int64_t> shape, bool norm=false) {
        auto t=std::make_shared<Tensor>(shape,kFloat32,kCPU);
        for(int64_t i=0;i<t->numel();++i) t->data<float>()[i]=(norm?1.0f:0.0f)+random(rng);
        model.assign_weight(key,t);
    };
    assign("embed_tokens.weight",{32,16}); assign("final_norm.weight",{16},true);
    for(int layer=0;layer<2;++layer) {
        const auto prefix="layers."+std::to_string(layer)+".";
        assign(prefix+"input_norm.weight",{16},true); assign(prefix+"post_norm.weight",{16},true);
        for(const std::string target:{"q","k","v","o"}) {
            const int dim=target=="k"||target=="v"?8:16;
            assign(prefix+"self_attn."+target+"_proj.weight",{16,dim});
            if(target!="o") assign(prefix+"self_attn."+target+"_proj.bias",{dim});
        }
        assign(prefix+"mlp.gate_proj.weight",{16,32}); assign(prefix+"mlp.up_proj.weight",{16,32});
        assign(prefix+"mlp.down_proj.weight",{32,16});
    }
    model.init_lora(2,4.0f,0.0f,false); model.freeze_base();
    auto params=model.get_lora_parameters();
    if(params.size()!=16) throw std::runtime_error("Expected 16 LoRA tensors");
    for(auto& p:params) for(int64_t i=0;i<p->numel();++i) p->data<float>()[i]=random(rng);
    auto ids=std::make_shared<Tensor>(std::vector<int64_t>{1,5},kInt32,kCPU);
    for(int i=0;i<5;++i) ids->data<int32_t>()[i]=(i*5+2)%32;
    auto weights=std::make_shared<Tensor>(std::vector<int64_t>{1,5,32},kFloat32,kCPU);
    for(int64_t i=0;i<weights->numel();++i) weights->data<float>()[i]=random(rng);
    auto output=model.forward(ids);
    if(output->shape()!=weights->shape()) throw std::runtime_error("Unexpected logits shape");
    auto value=[&](const TensorPtr& t) {
        double acc=0;
        for(int64_t i=0;i<t->numel();++i) acc+=double(t->data<float>()[i])*weights->data<float>()[i];
        return acc;
    };
    const auto objective=value(output);
    std::ofstream forward(std::string(argv[1])+".forward.csv");
    if(!forward) throw std::runtime_error("Cannot open forward CSV output");
    forward << "element,logit\n" << std::setprecision(12);
    for(int64_t i=0;i<output->numel();++i) forward << i << ',' << output->data<float>()[i] << '\n';
    // VJP seed corresponds exactly to the independent scalar dot product above.
    output->backward(weights);
    std::vector<std::vector<float>> grads;
    for(auto& p:params) {
        if(!p->grad()) throw std::runtime_error("Missing LoRA gradient");
        if(p->grad()->shape()!=p->shape() || p->grad()->dtype()!=kFloat32)
            throw std::runtime_error("LoRA gradient layout mismatch");
        grads.emplace_back(p->grad()->data<float>(),p->grad()->data<float>()+p->numel());
    }
    auto& engine=autograd::Engine::instance();
    engine.clear_graph(); engine.set_enabled(false);
    std::ofstream out(argv[1]);
    if(!out) throw std::runtime_error("Cannot open CSV output");
    out << std::setprecision(12) << "tensor,element,autograd,fd_eps_0.01,fd_eps_0.005\n";
    int count=0, failures=0; double error_max=0, fd_disagreement_max=0;
    for(size_t k=0;k<params.size();++k) {
        for(int64_t i=0;i<params[k]->numel();++i) {
            float& x=params[k]->data<float>()[i]; const float old=x;
            double fds[2]; int index=0;
            for(float eps:{0.01f,0.005f}) {
                x=old+eps; const double upper=value(model.forward(ids));
                x=old-eps; const double lower=value(model.forward(ids));
                x=old; fds[index++]=(upper-lower)/(2.0*eps);
            }
            const auto grad=grads[k][i];
            bool pass=std::isfinite(grad);
            for(double fd:fds) pass=pass && gradient_agrees(grad,fd);
            failures+=!pass; ++count;
            error_max=std::max(error_max,std::abs(grad-fds[1]));
            fd_disagreement_max=std::max(fd_disagreement_max,std::abs(fds[0]-fds[1]));
            out << k << ',' << i << ',' << grad << ',' << fds[0] << ',' << fds[1] << '\n';
        }
    }
    engine.set_enabled(true); engine.clear_graph();
    if(count!=448) throw std::runtime_error("Expected 448 finite-difference elements");
    std::cout << std::setprecision(12) << "{\"objective\":" << objective
              << ",\"gradient_entries\":" << count << ",\"failures\":" << failures
              << ",\"max_abs_error_fd_0.005\":" << error_max
              << ",\"fd_disagreement_max\":" << fd_disagreement_max << "}\n";
    return failures?1:0;
}
