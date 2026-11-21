#include "../core/training_session.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <cstring>
#include <cerrno>
#include <stdexcept>
#include <unistd.h>

using namespace ops;
using namespace ops::resume;
namespace fs=std::filesystem;
void require(bool p,const char* message) { if(!p) throw std::runtime_error(message); }
template<class F> void fails(F f) { bool thrown=false; try { f(); } catch(const std::exception&) { thrown=true; } require(thrown,"Expected operation to fail"); }
Json progress(int n) { return {{"completed_updates",n},{"total_steps",10},{"best_path",""}}; }
void update(const NamedParameters& ps,Adam& opt) {
    auto g=ones(ps[0].second->shape()); opt.step({ps[0].second},{g});
}
size_t payload_count(const std::string& run) {
    size_t count=0;
    for(const auto& e:fs::directory_iterator(fs::path(run)/"training_state"))
        if(fs::exists(e.path()/"state.safetensors")) ++count;
    return count;
}
int main() {
    try {
        std::string pattern=(fs::temp_directory_path()/"mft-session-tests-XXXXXX").string();
        const char* p=::mkdtemp(pattern.data()); require(p,"mkdtemp failed"); fs::path root=p;
        AdamConfig cfg; cfg.clip_grad_norm=0; cfg.amsgrad=true;
        auto param=ones({3}); NamedParameters named={{"layer.q.lora_A",param}}; Adam opt(cfg);
        const std::string run=(root/"run").string(); std::string committed;
        {
            TrainingSession s(run,false,"test",{},1,0,0); s.track_assets({});
            s.commit(named,opt,"{}",progress(0),"running");
            bool locked=false;
            try {TrainingSession busy(run,true,"test",{},1,0,0);}
            catch(const std::exception& e) { locked=std::string(e.what()).find("already active")!=std::string::npos; }
            require(locked,"Lock conflict did not reach flock");
            update(named,opt); s.event({{"type","update"},{"completed_updates",1}});
            s.commit(named,opt,"{}",progress(1),"running");
            committed=read_json(run+"/LATEST").at("current");
            s.set_commit_test_hook([](const std::string& stage){if(stage=="after_generation_rename") throw std::system_error(ENOSPC,std::generic_category(),"injected disk full");});
            update(named,opt); s.event({{"type","update"},{"completed_updates",2}});
            fails([&]{s.commit(named,opt,"{}",progress(2),"suspended");});
            require(read_json(run+"/LATEST").at("current")==committed,"Failed save replaced committed pointer");
        }
        auto abandoned_cache=fs::path(run)/"sessions"/"test-dead-session"/"artifacts"/"shards";
        fs::create_directories(abandoned_cache);
        {std::ofstream cache(abandoned_cache/"block.bin"); cache<<"disposable";}
        const auto latest=read_json(run+"/LATEST");
        const auto parent=fs::path(run)/"training_state";
        const auto current=parent/committed/"state.safetensors";
        const auto previous=parent/latest.at("previous").get<std::string>()/"state.safetensors";
        const auto offline=current.parent_path()/"temporarily-offline";
        fs::rename(current,offline);
        {
            TrainingSession s(run,true,"test",{},1,0,0); s.track_assets({});
            const float before=param->data<float>()[0];
            Adam fresh(cfg);
            bool unavailable=false;
            try { s.restore(named,fresh); }
            catch(const fs::filesystem_error& e) { unavailable=e.code()==std::errc::no_such_file_or_directory; }
            require(unavailable,"Unavailable current did not propagate its I/O error");
            require(param->data<float>()[0]==before && fresh.export_named_state(named).tensors.empty(),
                    "Failed restore changed model or optimizer");
            require(read_json(run+"/LATEST")==latest && fs::exists(offline) && fs::exists(previous),
                    "Failed restore changed the durable checkpoint");
            require(payload_count(run)==2 && fs::exists(abandoned_cache/"block.bin"),
                    "Failed restore reclaimed orphan payload or shard cache");
        }
        fs::rename(offline,current);
        float restored_value;
        {
            TrainingSession s(run,true,"test",{},1,0,0); s.track_assets({});
            require(payload_count(run)==3 && fs::exists(abandoned_cache),"Cleanup before restore validation");
            Adam fresh(cfg); auto state=s.restore(named,fresh);
            require(state && !state->fallback && state->generation==committed,"Did not retry the original checkpoint");
            require(payload_count(run)==2,"Orphan payload not reclaimed on restart");
            require(fs::exists(current) && fs::exists(previous),"Cleanup removed a committed generation");
            require(!fs::exists(abandoned_cache),"Killed session shard cache not reclaimed");
            require(state && state->trainer.at("completed_updates")==1,"Selected orphan/failed generation");
            restored_value=param->data<float>()[0];
            update(named,fresh); s.event({{"type","update"},{"completed_updates",2}});
            s.commit(named,fresh,"{}",progress(2),"suspended");
            require(payload_count(run)==2,"Payload retention exceeded current plus previous");
            auto latest=read_json(run+"/LATEST");
            auto cur=latest.at("current").get<std::string>();
            std::ofstream bad(run+"/training_state/"+cur+"/state.safetensors",std::ios::trunc); bad<<"broken";
        }
        {
            TrainingSession s(run,true,"test",{},1,0,0); s.track_assets({});
            Adam fresh(cfg); auto state=s.restore(named,fresh);
            require(state && state->fallback && state->trainer.at("completed_updates")==1,"Did not fallback to recorded committed predecessor");
            require(param->data<float>()[0]==restored_value,"Fallback weights mismatch");
            s.request_suspend(); require(s.suspend_requested(1),"Suspend request lost");
            s.commit(named,fresh,"{}",progress(1),"suspended");
            require(payload_count(run)==2,"Abandoned corrupt branch payload retained");
        }
        const std::string atomic=(root/"atomic").string();
        {
            TrainingSession s(atomic,false,"test",{},1,0,0); s.track_assets({});
            Adam fresh(cfg); s.commit(named,fresh,"{}",progress(0),"running");
            s.set_commit_test_hook([](const std::string& stage){if(stage=="after_pointer_rename") throw std::runtime_error("directory sync failed");});
            fails([&]{s.commit(named,fresh,"{}",progress(0),"suspended");});
            s.set_commit_test_hook({});
            fails([&]{s.commit(named,fresh,"{}",progress(0),"suspended");});
        }
        const std::string single=(root/"single").string();
        {
            TrainingSession s(single,false,"test",{},1,0,0); s.track_assets({});
            s.commit(named,opt,"{}",progress(0),"suspended");
        }
        const auto only=read_json(single+"/LATEST");
        require(only.at("previous")=="","Single generation fixture has a predecessor");
        const auto only_dir=fs::path(single)/"training_state"/only.at("current").get<std::string>();
        fs::rename(only_dir/"state.json",only_dir/"offline");
        {
            TrainingSession s(single,true,"test",{},1,0,0); s.track_assets({});
            Adam fresh(cfg); bool unavailable=false;
            try { s.restore(named,fresh); }
            catch(const fs::filesystem_error& e) { unavailable=e.code()==std::errc::no_such_file_or_directory; }
            require(unavailable,"Missing sole candidate did not propagate its I/O error");
        }
        fs::rename(only_dir/"offline",only_dir/"state.json");
        {std::ofstream bad(only_dir/"state.json",std::ios::trunc); bad<<"broken";}
        {
            TrainingSession s(single,true,"test",{},1,0,0); s.track_assets({});
            Adam fresh(cfg); bool exhausted=false;
            try { s.restore(named,fresh); }
            catch(const CheckpointsExhausted& e) { exhausted=e.exit_code==65; }
            require(exhausted && read_json(single+"/LATEST")==only,"Invalid sole candidate did not exhaust in place");
        }
        const std::string asset=(root/"asset.json").string(); write_json_durable(asset,{{"x",1}});
        const std::string identity=(root/"identity").string();
        {
            TrainingSession s(identity,false,"test",{},1,0,0); s.track_assets({asset});
            Adam fresh(cfg); s.commit(named,fresh,"{}",progress(0),"running");
        }
        write_json_durable(asset,{{"changed",2}});
        {
            TrainingSession s(identity,true,"test",{},1,0,0);
            fails([&]{s.track_assets({asset});});
        }
        const std::string tensorpath=(root/"roundtrip.safetensors").string();
        write_tensor_file(tensorpath,{{"x",param}}); auto roundtrip=read_tensor_file(tensorpath);
        require(!std::memcmp(param->data<float>(),roundtrip.at("x")->data<float>(),3*sizeof(float)),"Tensor roundtrip mismatch");
        {std::ofstream append(tensorpath,std::ios::app|std::ios::binary); append<<"x";}
        fails([&]{read_tensor_file(tensorpath);});
        fs::remove_all(root);
        std::cout<<"PASS: lock, transaction rollback, ENOSPC injection, orphan exclusion, fallback, suspend, post-pointer poison, asset mismatch, tensor roundtrip/corruption\n";
        return 0;
    } catch(const std::exception& e) {std::cerr<<"FAIL: "<<e.what()<<"\n"; return 1;}
}
