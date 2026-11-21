#include "../core/training_session.h"
#include "../core/phase_probe.h"
#include <filesystem>
#include <iostream>
#include <unistd.h>

int main() {
    namespace fs=std::filesystem;
    using namespace ops;
    using namespace ops::resume;
    auto path=fs::temp_directory_path()/("mft-phase-"+std::to_string(getpid())+".safetensors");
    try {
        auto tensor=std::make_shared<Tensor>(std::vector<int64_t>{16},kFloat32,kCPU);
        for(int i=0;i<16;++i) tensor->data<float>()[i]=static_cast<float>(i);
        PhaseProbe save;
        write_tensor_file(path.string(),{{"weights",tensor}});
        auto s=save.finish();
        if(s.at("logical_write_bytes")!=fs::file_size(path) || s.at("fsync_calls")!=1)
            throw std::runtime_error("Save byte/fsync accounting mismatch");
        PhaseProbe restore;
        auto loaded=read_tensor_file(path.string());
        auto r=restore.finish();
        if(r.at("logical_read_bytes")!=fs::file_size(path) || r.at("logical_write_bytes")!=0)
            throw std::runtime_error("Read byte accounting mismatch");
        for(int i=0;i<16;++i) if(loaded.at("weights")->data<float>()[i]!=i) throw std::runtime_error("Payload mismatch");
        for(const auto& m:{s,r}) {
            if(m.at("wall_sec").get<double>()<0 || m.at("rss_sample_count").get<int>()<2 ||
               m.at("peak_sampled_rss_bytes").is_null() || m.at("peak_sampled_rss_bytes")<m.at("rss_start_bytes") ||
               m.at("peak_sampled_rss_bytes")<m.at("rss_end_bytes"))
                throw std::runtime_error("Invalid phase measurement");
        }
        for(int i=0;i<100;++i) {
            PhaseProbe p;
            std::vector<char> touched(1024*1024, static_cast<char>(i));
            auto m=p.finish();
            if(m.at("peak_sampled_rss_bytes")<m.at("rss_start_bytes") ||
               m.at("peak_sampled_rss_bytes")<m.at("rss_end_bytes"))
                throw std::runtime_error("RSS endpoints excluded from peak");
        }
        if(save.finish()!=s) throw std::runtime_error("finish must be idempotent");
        fs::remove(path);
        std::cout<<"PASS exact logical I/O, fsync, RSS and idempotent measurement\n";
        return 0;
    } catch(const std::exception& e) { fs::remove(path); std::cerr<<e.what()<<'\n'; return 1; }
}
