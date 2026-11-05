#include "memory_efficient_attention.h"
#include "autograd_engine.h"
#include "checkpoint.h"
#include "ops.h"
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

using namespace ops;
void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
float maxdiff(const TensorPtr& a, const TensorPtr& b) {
    require(a && b && a->shape()==b->shape(),"missing tensor/shape mismatch");
    auto ac=a, bc=b;
    float err=0;
    for (int64_t i=0;i<a->numel();++i) {
        require(std::isfinite(ac->data<float>()[i]) && std::isfinite(bc->data<float>()[i]),"nonfinite value");
        err=std::max(err,std::abs(ac->data<float>()[i]-bc->data<float>()[i]));
    }
    return err;
}
TensorPtr value(int seed, bool strided) {
    auto t=zeros(strided ? std::vector<int64_t>{2,3,2,4} : std::vector<int64_t>{2,2,3,4});
    for (int64_t i=0;i<t->numel();++i) t->data<float>()[i]=std::sin(float(i+seed))*.3f;
    if (strided) t=t->transpose(1,2);
    t->set_requires_grad(true);
    return t;
}
int main() try {
    auto& engine=autograd::Engine::instance();
    for (int mode=0;mode<5;++mode) for (bool strided : {false,true}) {
        engine.clear_graph();
        auto q=value(1,strided),k=value(2,strided),v=value(3,strided);
        auto qr=q->clone(),kr=k->clone(),vr=v->clone();
        qr->set_requires_grad(true);kr->set_requires_grad(true);vr->set_requires_grad(true);
        TensorPtr mask=nullptr,pad=nullptr;
        MemoryEfficientAttentionConfig cfg; cfg.use_causal_mask=mode!=0;cfg.scale=.37f;
        if (mode>=2) {
            mask=zeros({3,3});
            for(int i=0;i<3;++i) for(int j=0;j<3;++j)
                mask->data<float>()[i*3+j]=(j>i || (mode==3 && j<i-1)) ? -1e9f : 0;
        }
        if (mode>=3) { pad=zeros({2,1,1,3}); pad->data<float>()[2]=-1e9f; }
        if(mode==4) { mask=mask->transpose(0,1); pad=reshape(pad,{2,3}); }
        auto out=memory_efficient_attention(q,k,v,mask,pad,cfg);
        auto node=engine.get_node(out);
        require(node && node->next_edges.size()==3,"not a direct streaming autograd node");
        auto scores=mul(matmul(qr,transpose(kr,2,3)),cfg.scale);
        if(mask) scores=add(scores,mask);
        else if(cfg.use_causal_mask) scores=add(scores,create_causal_mask(3,kFloat32,kCPU));
        if(pad) scores=add(scores,reshape(pad,{2,1,1,3}));
        auto ref=matmul(softmax(scores,-1),vr);
        auto go=value(7,false); go->set_requires_grad(false);
        engine.run_backward({out,ref},{go,go});
        require(maxdiff(out,ref)<2e-6f,"forward mismatch");
        require(maxdiff(q->grad(),qr->grad())<2e-6f,"Q gradient mismatch");
        require(maxdiff(k->grad(),kr->grad())<2e-6f,"K gradient mismatch");
        require(maxdiff(v->grad(),vr->grad())<2e-6f,"V gradient mismatch");
        if(!strided && mode==0) {
            engine.clear_graph(); engine.set_enabled(false);
            for(auto t : {q,k,v}) for(int64_t i=0;i<t->numel();++i) {
                float old=t->data<float>()[i], vals[2];
                for(int sign=0;sign<2;++sign) {
                    t->data<float>()[i]=old+(sign?-.001f:.001f);
                    auto f=memory_efficient_attention(q,k,v,mask,pad,cfg); double dot=0;
                    for(int64_t j=0;j<f->numel();++j) dot+=f->data<float>()[j]*go->data<float>()[j];
                    vals[sign]=static_cast<float>(dot);
                }
                t->data<float>()[i]=old;
                require(std::abs((vals[0]-vals[1])/.002f-t->grad()->data<float>()[i])<1e-4f,"finite difference mismatch");
            }
            engine.set_enabled(true);
        }
    }
    // Exercise primitive composition directly; model projections can otherwise
    // hide a missing dependency registration in the streaming operator itself.
    for (bool explicit_captures : {false,true}) {
        engine.clear_graph();
        auto q=value(1,false),k=value(2,false),v=value(3,false);
        auto qr=q->clone(),kr=k->clone(),vr=v->clone();
        qr->set_requires_grad(true);kr->set_requires_grad(true);vr->set_requires_grad(true);
        auto wrapped=checkpoint([k,v](const TensorPtr& input){return memory_efficient_attention(input,k,v);},q,
                                explicit_captures ? std::vector<TensorPtr>{k,v} : std::vector<TensorPtr>{});
        auto ref=memory_efficient_attention(qr,kr,vr);
        auto grad=value(9,false);grad->set_requires_grad(false);
        engine.run_backward({wrapped,ref},{grad,grad});
        require(maxdiff(q->grad(),qr->grad())<1e-7f,"checkpoint attention Q dependency lost");
        require(maxdiff(k->grad(),kr->grad())<1e-7f,"checkpoint attention K dependency lost");
        require(maxdiff(v->grad(),vr->grad())<1e-7f,"checkpoint attention V dependency lost");
    }
    engine.clear_graph();
    auto q=value(1,false),k=value(2,false),v=value(3,false),mask=zeros({3,3});
    for(int j=0;j<3;++j)mask->data<float>()[j]=-std::numeric_limits<float>::infinity();
    auto out=memory_efficient_attention(q,k,v,mask);
    auto go=zeros(out->shape());
    for(int b=0;b<2;++b)for(int h=0;h<2;++h)for(int d=0;d<4;++d) {
        int idx=((b*2+h)*3)*4+d; require(out->data<float>()[idx]==0,"fully masked output");go->data<float>()[idx]=1;
    }
    engine.run_backward({out},{go});
    for(auto t:{q,k,v})require(maxdiff(t->grad(),zeros(t->shape()))==0,"fully masked gradient");
    bool rejected=false;
    try { memory_efficient_attention(q,k,v,nullptr,zeros({2,2,1,3}),{}); }
    catch(const std::invalid_argument&){rejected=true;}
    require(rejected,"bad padding accepted");
    std::cout<<"PASS: 10 forward/backward cases, 144 finite differences, masks and direct streaming node\n";
} catch(const std::exception& e) { std::cerr<<e.what()<<'\n';return 1; }
