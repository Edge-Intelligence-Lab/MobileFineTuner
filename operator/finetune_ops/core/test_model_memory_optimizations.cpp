#include "../graph/gemma_lora_injector.h"
#include "../graph/gemma_model.h"
#include "../graph/qwen_model.h"
#include "../optim/adam.h"
#include "autograd_engine.h"
#include "lm_loss.h"
#include "ops.h"
#include <chrono>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <type_traits>

using namespace ops;
namespace {
void require(bool v,const std::string& message){if(!v)throw std::runtime_error(message);}
float compare(const TensorPtr& a,const TensorPtr& b) {
    require(a && b && a->shape()==b->shape(),"missing gradient or tensor shape");
    float error=0;
    for(int64_t i=0;i<a->numel();++i) {
        require(std::isfinite(a->data<float>()[i]) && std::isfinite(b->data<float>()[i]),"non-finite tensor");
        error=std::max(error,std::abs(a->data<float>()[i]-b->data<float>()[i]));
    }
    return error;
}
template<class Model> void base(Model& model) {
    auto params=model.parameters();
    for(size_t n=0;n<params.size();++n) {
        auto& p=params[n];p->set_requires_grad(false);
        for(int64_t i=0;i<p->numel();++i)
            p->template data<float>()[i]=p->ndim()==1 ? 1.0f : .07f*std::sin(float(i+n+1));
    }
}
void adapters(const std::vector<TensorPtr>& params) {
    for(size_t n=0;n<params.size();++n) for(int64_t i=0;i<params[n]->numel();++i)
        params[n]->data<float>()[i]=.02f*std::cos(float(i+n+1));
}
struct Observation {
    float loss;
    std::vector<TensorPtr> grads, params;
    size_t graph_nodes = 0;
};
template <class Model>
std::vector<Observation> run(Model& model, int mode, const std::filesystem::path& directory,
                             const std::vector<Observation>* aligned = nullptr,
                             bool streamed_head = false) {
    auto params=model.get_lora_parameters();adapters(params);
    model.set_memory_efficient_attention((mode&1)!=0);
    model.set_activation_checkpointing((mode&2)!=0);
    const auto frozen_count = model.parameters().size();
    if(mode&4) {
        sharding::ShardConfig cfg;cfg.offload_dir=directory.string();cfg.max_resident_bytes=8192;
        model.enable_parameter_sharding(cfg);
        require(model.sharding_stats().resident_bytes==0,"initial offload did not release managed owners");
        require(model.sharding_stats().entries == frozen_count,
                "offload inventory omitted frozen weights");
    }
    auto ids=zeros({2,5},kInt32),labels=zeros({2,5},kInt32),mask=ones({2,5});
    for(int i=0;i<10;++i) {ids->data<int32_t>()[i]=(i+1)%31;labels->data<int32_t>()[i]=(i+2)%31;}
    labels->data<int32_t>()[4]=-100;mask->data<float>()[4]=0;
    labels->data<int32_t>()[8]=-100;labels->data<int32_t>()[9]=-100;
    mask->data<float>()[8]=mask->data<float>()[9]=0;
    Adam optimizer(AdamConfig(.001f,.9f,.999f,1e-6f));
    std::vector<Observation> observations;
    for(int step=0;step<3;++step) {
        autograd::Engine::instance().clear_graph();
        // A separate run checks derivatives at identical parameter values;
        // the ordinary run below still checks independently evolving updates.
        if(aligned && step>0) for(size_t p=0;p<params.size();++p)
            std::copy((*aligned)[step-1].params[p]->data<float>(),
                      (*aligned)[step-1].params[p]->data<float>()+params[p]->numel(),params[p]->template data<float>());
        for(auto& p:params)p->zero_grad();
        TensorPtr loss;
        if constexpr (std::is_same<Model, QwenModel>::value) {
            if (streamed_head)
                loss = model.loss_from_hidden(model.forward_hidden(ids, mask), labels, true);
        }
        if (!loss)
            loss = lm_cross_entropy(model.forward(ids, mask), labels);
        auto graph = autograd::Engine::instance().take_state();
        const auto graph_nodes = graph.nodes.size();
        autograd::Engine::instance().restore_state(std::move(graph));
        loss->backward();
        Observation result;result.loss=loss->template data<float>()[0];
        result.graph_nodes = graph_nodes;
        std::vector<TensorPtr> grads;
        for(auto& p:params){require(bool(p->grad()),"LoRA gradient missing");grads.push_back(p->grad());result.grads.push_back(p->grad()->clone());}
        optimizer.step(params,grads);
        for(auto& p:params)result.params.push_back(p->clone());
        observations.push_back(result);
    }
    autograd::Engine::instance().clear_graph();
    if(mode&4) {
        auto stats=model.sharding_stats();
        require(stats.loads>stats.entries && stats.evictions>stats.entries && stats.bytes_read>0 && stats.bytes_written>0,"no disk reload activity");
        require(stats.writes==stats.entries,"unchanged frozen weights were unnecessarily rewritten");
        std::cout<<"I/O mode="<<mode<<" loads="<<stats.loads<<" read="<<stats.bytes_read<<" written="<<stats.bytes_written<<'\n';
    }
    return observations;
}
void check(const std::vector<Observation>& ref,const std::vector<Observation>& candidate,const std::string& name,
           bool check_gradients=true) {
    float max_grad=0,max_param=0,max_scaled_grad=0;
    require(ref.size()==candidate.size(),"update count");
    for(size_t s=0;s<ref.size();++s) {
        require(std::abs(ref[s].loss-candidate[s].loss)<2e-5f,name+" loss mismatch");
        require(ref[s].params.size()==candidate[s].params.size(),"parameter count");
        for(size_t p=0;p<ref[s].params.size();++p) {
            const float error=compare(ref[s].grads[p],candidate[s].grads[p]);
            max_grad=std::max(max_grad,error);
            // FP32 row kernels and GEMM reduce in different orders. Check
            // each coordinate so large tensor entries cannot hide small ones.
            for(int64_t i=0;i<ref[s].grads[p]->numel();++i) {
                const float a=ref[s].grads[p]->data<float>()[i];
                const float b=candidate[s].grads[p]->data<float>()[i];
                const float tolerance=2e-5f+1e-4f*std::max(std::abs(a),std::abs(b));
                max_scaled_grad=std::max(max_scaled_grad,std::abs(a-b)/tolerance);
                if (check_gradients && std::abs(a-b)>tolerance)
                    std::cerr<<name<<" gradient mismatch step="<<s+1<<" tensor="<<p<<" index="<<i
                             <<" reference="<<a<<" candidate="<<b<<" tolerance="<<tolerance<<'\n';
            }
            max_param=std::max(max_param,compare(ref[s].params[p],candidate[s].params[p]));
        }
    }
    std::cout<<name<<" grad_max="<<max_grad<<" param_max="<<max_param<<" scaled_grad="<<max_scaled_grad<<'\n';
    require((!check_gradients || max_scaled_grad<1) && max_param<2e-5f,name+" update parity mismatch");
}
}
int main() try {
    auto dir=std::filesystem::temp_directory_path()/ ("mft_model_memory_"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(dir);
    QwenConfig q;q.vocab_size=32;q.hidden_size=8;q.intermediate_size=16;q.num_hidden_layers=2;q.num_attention_heads=2;q.num_key_value_heads=1;
    GemmaTextConfig g;g.vocab_size=32;g.hidden_size=8;g.intermediate_size=16;g.num_hidden_layers=2;g.num_attention_heads=2;g.num_key_value_heads=1;g.head_dim=4;
    g.query_pre_attn_scalar=9;g.sliding_window=2;g.layer_types={"sliding_attention","full_attention"};
    QwenConfig q3 = q;
    q3.model_type = "qwen3";
    q3.head_dim = 6;
    q3.attention_bias = false;
    q3.tie_word_embeddings = false;
    std::vector<Observation> qref, gref, q3ref;
    std::vector<Observation> hybrid_refs[3];
    for(int mode=0;mode<8;++mode) {
        QwenModel qm(q);base(qm);qm.init_lora(2,4,0,false,42);
        auto qr=run(qm,mode,dir);if(!mode)qref=qr;else check(qref,qr,"Qwen trajectory mode="+std::to_string(mode),!(mode&1));
        GemmaModel gm(g);base(gm);GemmaLoraInjector injector;
        auto spec=GemmaLoraSpec::full_attn_mlp();spec.rank=2;spec.alpha=4;spec.dropout=0;injector.inject(gm,spec);
        auto gr=run(gm,mode,dir);if(!mode)gref=gr;else check(gref,gr,"Gemma trajectory mode="+std::to_string(mode),!(mode&1));
        QwenModel q3m(q3);
        base(q3m);
        q3m.init_lora(2, 4, 0, std::vector<std::string>{"all-linear"}, 42);
        auto q3r = run(q3m, mode, dir);
        if (!mode)
            q3ref = q3r;
        else
            check(q3ref, q3r, "Qwen3 trajectory mode=" + std::to_string(mode), !(mode & 1));
        for (int variant = 0; variant < 3; ++variant) {
            QwenConfig h = q;
            h.model_type = "qwen3_5_text";
            h.attention_bias = false;
            h.head_dim = 8;
            h.tie_word_embeddings = variant != 1;
            h.linear_num_key_heads = 1;
            h.linear_num_value_heads = 2;
            h.linear_key_head_dim = 3;
            h.linear_value_head_dim = 4;
            h.linear_conv_kernel_dim = 3;
            h.partial_rotary_factor = 0.5f;
            h.layer_types = variant == 0
                                ? std::vector<std::string>{"linear_attention", "full_attention"}
                                : std::vector<std::string>{"full_attention", "linear_attention",
                                                           "linear_attention"};
            h.num_hidden_layers = int(h.layer_types.size());
            const std::vector<std::string> targets = variant == 2
                                                         ? std::vector<std::string>{"q_proj"}
                                                         : std::vector<std::string>{"all-linear"};
            QwenModel hybrid(h);
            base(hybrid);
            hybrid.init_lora(2, 4, 0, targets, 42);
            auto candidate = run(hybrid, mode, dir);
            const std::string name =
                "Qwen3.5 variant=" + std::to_string(variant) + " mode=" + std::to_string(mode);
            if (!mode)
                hybrid_refs[variant] = candidate;
            else
                check(hybrid_refs[variant], candidate, name, !(mode & 1));
            if (mode & 2) {
                require(candidate.front().graph_nodes < hybrid_refs[variant].front().graph_nodes,
                        "Hybrid checkpointing did not reduce retained autograd nodes");
                std::cout << name << " graph_nodes=" << candidate.front().graph_nodes
                          << " baseline_nodes=" << hybrid_refs[variant].front().graph_nodes << '\n';
            }
            require(hybrid.named_lora_parameters().size() == hybrid.get_lora_parameters().size(),
                    "Offload changed hybrid adapter inventory");
            hybrid.save_lora_adapter(
                (dir / ("adapter_" + std::to_string(variant) + "_" + std::to_string(mode)))
                    .string());
            QwenModel streamed(h);
            base(streamed);
            streamed.init_lora(2, 4, 0, targets, 42);
            check(hybrid_refs[variant], run(streamed, mode, dir, nullptr, true),
                  name + " streamed-head", !(mode & 1));
            if (mode & 1) {
                QwenModel aligned(h);
                base(aligned);
                aligned.init_lora(2, 4, 0, targets, 42);
                check(hybrid_refs[variant], run(aligned, mode, dir, &hybrid_refs[variant]),
                      name + " identical-input");
            }
        }
        if(mode&1) {
            QwenModel qa(q);base(qa);qa.init_lora(2,4,0,false,42);
            check(qref,run(qa,mode,dir,&qref),"Qwen identical-input mode="+std::to_string(mode));
            GemmaModel ga(g);base(ga);injector.inject(ga,spec);
            check(gref,run(ga,mode,dir,&gref),"Gemma identical-input mode="+std::to_string(mode));
            QwenModel q3a(q3);
            base(q3a);
            q3a.init_lora(2, 4, 0, std::vector<std::string>{"all-linear"}, 42);
            check(q3ref, run(q3a, mode, dir, &q3ref),
                  "Qwen3 identical-input mode=" + std::to_string(mode));
        }
    }
    std::filesystem::remove_all(dir);
    std::cout << "PASS: Qwen2/Qwen3/Gemma and three Qwen3.5 variants, eight combinations, three "
                 "Adam updates and "
                 "identical-input gradient checks\n";
} catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
