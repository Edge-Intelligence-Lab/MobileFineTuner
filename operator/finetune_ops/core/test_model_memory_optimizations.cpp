#include "../graph/qwen_model.h"
#include "../graph/gemma_model.h"
#include "../graph/gemma_lora_injector.h"
#include "../optim/adam.h"
#include "autograd_engine.h"
#include "lm_loss.h"
#include "ops.h"
#include <chrono>
#include <filesystem>
#include <iostream>
#include <stdexcept>

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
struct Observation {float loss;std::vector<TensorPtr> grads,params;};
template<class Model> std::vector<Observation> run(Model& model,int mode,const std::filesystem::path& directory,
                                                  const std::vector<Observation>* aligned=nullptr) {
    auto params=model.get_lora_parameters();adapters(params);
    model.set_memory_efficient_attention((mode&1)!=0);
    model.set_activation_checkpointing((mode&2)!=0);
    if(mode&4) {
        sharding::ShardConfig cfg;cfg.offload_dir=directory.string();cfg.max_resident_bytes=8192;
        model.enable_parameter_sharding(cfg);
        require(model.sharding_stats().resident_bytes==0,"initial offload did not release managed owners");
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
        auto logits=model.forward(ids,mask);
        auto loss=lm_cross_entropy(logits,labels);
        loss->backward();
        Observation result;result.loss=loss->template data<float>()[0];
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
    std::vector<Observation> qref,gref;
    for(int mode=0;mode<8;++mode) {
        QwenModel qm(q);base(qm);qm.init_lora(2,4,0,false,42);
        auto qr=run(qm,mode,dir);if(!mode)qref=qr;else check(qref,qr,"Qwen trajectory mode="+std::to_string(mode),!(mode&1));
        GemmaModel gm(g);base(gm);GemmaLoraInjector injector;
        auto spec=GemmaLoraSpec::full_attn_mlp();spec.rank=2;spec.alpha=4;spec.dropout=0;injector.inject(gm,spec);
        auto gr=run(gm,mode,dir);if(!mode)gref=gr;else check(gref,gr,"Gemma trajectory mode="+std::to_string(mode),!(mode&1));
        if(mode&1) {
            QwenModel qa(q);base(qa);qa.init_lora(2,4,0,false,42);
            check(qref,run(qa,mode,dir,&qref),"Qwen identical-input mode="+std::to_string(mode));
            GemmaModel ga(g);base(ga);injector.inject(ga,spec);
            check(gref,run(ga,mode,dir,&gref),"Gemma identical-input mode="+std::to_string(mode));
        }
    }
    std::filesystem::remove_all(dir);
    std::cout<<"PASS: Qwen/Gemma, eight combinations, three Adam updates and identical-input gradient checks\n";
} catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
