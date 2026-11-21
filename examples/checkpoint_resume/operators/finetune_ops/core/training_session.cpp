#include "training_session.h"
#include <algorithm>
#include <cerrno>
#include <cmath>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <set>
#include <stdexcept>
#include <system_error>
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>
#ifdef __ANDROID__
#include <sys/prctl.h>
#endif
#ifdef __APPLE__
#include <mach-o/dyld.h>
#endif

namespace ops::resume {
namespace fs = std::filesystem;
namespace {
struct InvalidGeneration : std::runtime_error { using std::runtime_error::runtime_error; };

// Only wrap validators operating on already-loaded bytes/state, never asset or file I/O.
template<class F> auto validate_content(F&& f) {
    try { return f(); }
    catch (const std::system_error&) { throw; }
    catch (const std::runtime_error& e) { throw InvalidGeneration(e.what()); }
}
volatile std::sig_atomic_t signal_suspend = 0;
void on_signal(int) { signal_suspend = 1; }
struct Fd {
    int value;
    explicit Fd(int v) : value(v) { if (v < 0) throw std::system_error(errno, std::generic_category(), "open checkpoint file"); }
    ~Fd() { if (value >= 0) ::close(value); }
    Fd(const Fd&) = delete;
    Fd& operator=(const Fd&) = delete;
};
void sync_fd(int fd) {
    if (::fsync(fd) != 0) throw std::system_error(errno, std::generic_category(), "checkpoint fsync");
    ++logical_io().fsyncs;
}
void sync_dir(const fs::path& p) { Fd fd(::open(p.c_str(), O_RDONLY | O_CLOEXEC)); sync_fd(fd.value); }
void mkdir_sync(const fs::path& p) {
    if (fs::exists(p)) { if (!fs::is_directory(p)) throw std::runtime_error("Not a directory: " + p.string()); return; }
    if (!p.parent_path().empty()) mkdir_sync(p.parent_path());
    fs::create_directory(p); sync_dir(p); sync_dir(p.parent_path());
}
void write_all(int fd, const void* data, size_t bytes) {
    const char* p = static_cast<const char*>(data);
    while (bytes) {
        auto n = ::write(fd, p, std::min<size_t>(bytes, 1024 * 1024));
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) throw std::system_error(n < 0 ? errno : EIO, std::generic_category(), "checkpoint write");
        logical_io().write_bytes+=static_cast<uint64_t>(n); ++logical_io().writes;
        p += n; bytes -= static_cast<size_t>(n);
    }
}
void read_all(std::ifstream& in, void* p, size_t n) {
    in.read(static_cast<char*>(p), static_cast<std::streamsize>(n));
    if (!in) throw std::runtime_error("Truncated checkpoint tensor file");
    logical_io().read_bytes+=n; ++logical_io().reads;
}
void check_component(const std::string& s) {
    if (s.empty() || s == "." || s == ".." || s.find('/') != s.npos || s.find('\\') != s.npos)
        throw std::runtime_error("Invalid checkpoint generation name");
}
void prune_payloads(const fs::path& root,const std::string& current,const std::string& previous) {
    check_component(current); if(!previous.empty()) check_component(previous);
    const auto parent=root/"training_state";
    if(!fs::exists(parent)) return;
    for(const auto& entry:fs::directory_iterator(parent)) {
        const auto name=entry.path().filename().string();
        if(!entry.is_directory() || name.rfind("g-",0)!=0 || name==current || name==previous) continue;
        // Manifests/events remain for metrics lineage; uncommitted tensor payloads
        // and failed-write leftovers are never promoted into recovery authority.
        for(const auto* file:{"state.safetensors","state.json"}) {
            std::error_code ec; fs::remove(entry.path()/file,ec);
            if(ec) std::cerr<<"Checkpoint cleanup deferred: "<<entry.path()/file<<": "<<ec.message()<<"\n";
        }
    }
}
void prune_shard_cache(const fs::path& session) {
    std::error_code ec;
    fs::remove_all(session/"artifacts"/"shards",ec);
    if(ec) std::cerr<<"Disposable shard cache cleanup deferred: "<<ec.message()<<"\n";
}
Json stamp(const std::string& filename) {
    fs::path p = fs::canonical(filename);
    if (!fs::is_regular_file(p)) throw std::runtime_error("Checkpoint asset is not a regular file: " + filename);
    return {{"path",p.string()}, {"bytes",fs::file_size(p)}, {"mtime",fs::last_write_time(p).time_since_epoch().count()}};
}
std::string running_executable() {
    // argv[0] may be only a PATH command or alias; save the running image,
    // not a future PATH lookup that could resolve to a different build.
#ifdef __APPLE__
    uint32_t size = 0;
    _NSGetExecutablePath(nullptr, &size);
    std::vector<char> path(size);
    if (_NSGetExecutablePath(path.data(), &size) != 0)
        throw std::runtime_error("Cannot resolve the running executable");
    return fs::canonical(path.data()).string();
#elif defined(__linux__)
    return fs::canonical("/proc/self/exe").string();
#else
    throw std::runtime_error("Exact resume requires a supported executable identity API");
#endif
}
Json thread_environment() {
    Json j = Json::object();
    for (const auto* key : {"OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","VECLIB_MAXIMUM_THREADS","MKL_NUM_THREADS","MFT_NUM_THREADS"}) {
        const char* v = std::getenv(key); j[key] = v ? Json(v) : Json(nullptr);
    }
    return j;
}
std::string unique_id() {
    auto t = std::chrono::system_clock::now().time_since_epoch();
    return std::to_string(std::chrono::duration_cast<std::chrono::nanoseconds>(t).count()) + "-" + std::to_string(::getpid());
}
void check_tensor(const TensorPtr& t) {
    if (!t || t->dtype() != kFloat32 || t->device() != kCPU || !t->is_contiguous() || t->numel() <= 0)
        throw std::runtime_error("Resume supports nonempty contiguous CPU FP32 tensors only");
    for (int64_t i=0;i<t->numel();++i) if (!std::isfinite(t->data<float>()[i])) throw std::runtime_error("Nonfinite checkpoint tensor");
}
void check_progress(const Json& j) {
    const auto step = j.at("completed_updates").get<int64_t>();
    const auto total = j.at("total_steps").get<int64_t>();
    if (step < 0 || total <= 0 || step > total) throw std::runtime_error("Invalid committed training progress");
}
Json validate_generation(const fs::path& root, const std::string& g, TensorMap& tensors) {
    check_component(g);
    fs::path p = root / "training_state" / g;
    Json manifest = read_json((p / "manifest.json").string());
    if (manifest.at("schema") != 1 || manifest.at("generation") != g) throw InvalidGeneration("Invalid generation manifest");
    for (const auto* name : {"state.json", "state.safetensors"})
        if (fs::file_size(p/name) != manifest.at("files").at(name).get<uint64_t>()) throw InvalidGeneration("Checkpoint file size mismatch");
    Json s = read_json((p/"state.json").string());
    if (s.at("schema") != 1) throw InvalidGeneration("Unknown checkpoint schema");
    validate_content([&] { check_progress(s.at("trainer")); });
    if (manifest.at("completed_updates") != s.at("trainer").at("completed_updates")) throw InvalidGeneration("Checkpoint progress mismatch");
    const auto dataset=Json::parse(s.at("dataset_state").get<std::string>());
    if(!dataset.is_object()) throw InvalidGeneration("Invalid dataset state");
    if (manifest.contains("best_file") && !manifest.at("best_file").is_null()) {
        const auto& b = manifest.at("best_file");
        if (stamp(b.at("path").get<std::string>()) != b) throw std::runtime_error("Best adapter changed or missing");
    }
    tensors = read_tensor_file((p/"state.safetensors").string());
    s["manifest"] = manifest;
    return s;
}
}  // namespace

Json read_json(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Cannot read " + path);
    // Separate short/failed reads from parse failures on a complete candidate.
    std::string bytes(fs::file_size(path), '\0');
    read_all(in,bytes.data(),bytes.size());
    return Json::parse(bytes);
}
void write_json_durable(const std::string& path, const Json& value) {
    const std::string text = value.dump(2) + "\n";
    Fd fd(::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0600));
    write_all(fd.value, text.data(), text.size()); sync_fd(fd.value);
}
void write_tensor_file(const std::string& path, const TensorMap& tensors) {
    if (tensors.empty()) throw std::runtime_error("Empty checkpoint tensor set");
    Json h = Json::object(); uint64_t offset = 0;
    for (const auto& [name,t] : tensors) {
        check_tensor(t);
        if (name.empty() || name == "__metadata__") throw std::runtime_error("Invalid tensor name");
        uint64_t size = static_cast<uint64_t>(t->numel()) * sizeof(float);
        h[name] = {{"dtype","F32"},{"shape",t->shape()},{"data_offsets",{offset,offset+size}}}; offset += size;
    }
    std::string header = h.dump(); while (header.size()%8) header += ' ';
    uint64_t len = header.size(); unsigned char le[8];
    for (int i=0;i<8;++i) le[i] = static_cast<unsigned char>((len >> (8*i)) & 255);
    Fd fd(::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC,0600));
    write_all(fd.value,le,8); write_all(fd.value,header.data(),header.size());
    for (const auto& [name,t] : tensors) write_all(fd.value,t->data<float>(),static_cast<size_t>(t->numel())*sizeof(float));
    sync_fd(fd.value);
}
TensorMap read_tensor_file(const std::string& path) {
    std::ifstream in(path,std::ios::binary);
    if (!in) throw std::runtime_error("Cannot read checkpoint tensors");
    const uint64_t bytes=fs::file_size(path); unsigned char le[8]; read_all(in,le,8);
    uint64_t len=0; for(int i=0;i<8;++i) len |= static_cast<uint64_t>(le[i])<<(8*i);
    if (bytes<8 || len>bytes-8 || len>64*1024*1024 || !len) throw InvalidGeneration("Invalid tensor header size");
    std::string text(len,' '); read_all(in,text.data(),len); Json h=Json::parse(text);
    if (!h.is_object() || h.empty()) throw InvalidGeneration("Invalid tensor header");
    uint64_t expected=0; TensorMap out;
    for (auto it=h.begin();it!=h.end();++it) {
        if (it.key()=="__metadata__") continue;
        const auto& v=it.value(); auto shape=v.at("shape").get<std::vector<int64_t>>();
        auto offsets=v.at("data_offsets").get<std::vector<uint64_t>>(); uint64_t n=1;
        for(auto d:shape) {
            if(d<=0 || static_cast<uint64_t>(d)>(bytes-8-len)/sizeof(float)/n) throw InvalidGeneration("Invalid tensor shape");
            n*=static_cast<uint64_t>(d);
        }
        if(v.at("dtype")!="F32" || offsets.size()!=2 || offsets[0]!=expected || offsets[1]!=expected+n*sizeof(float) || offsets[1]>bytes-8-len)
            throw InvalidGeneration("Invalid tensor offsets/type");
        auto t=std::make_shared<Tensor>(shape,kFloat32,kCPU);
        read_all(in,t->data<float>(),n*sizeof(float)); validate_content([&] { check_tensor(t); }); out[it.key()]=t; expected=offsets[1];
    }
    if(expected!=bytes-8-len || out.empty()) throw InvalidGeneration("Unexpected checkpoint payload length");
    return out;
}

TrainingSession::TrainingSession(const std::string& run_dir, bool restoring,
    const std::string& program,const std::vector<std::string>& arguments,int every,int stop,double seconds)
    : run_dir_(fs::absolute(run_dir).lexically_normal().string()),program_(program),restoring_(restoring),
      checkpoint_every_(every),suspend_after_(stop),slice_seconds_(seconds),started_(std::chrono::steady_clock::now()) {
    if(every<1 || stop<0 || seconds<0 || !std::isfinite(seconds)) throw std::runtime_error("Invalid resume controls");
    try {
        if(restoring && !fs::exists(fs::path(run_dir_)/"LATEST")) throw std::runtime_error("No committed LATEST to resume");
        mkdir_sync(run_dir_);
        lock_fd_=::open((fs::path(run_dir_)/"run.lock").c_str(),O_RDWR|O_CREAT|O_CLOEXEC,0600);
        if(lock_fd_<0 || ::flock(lock_fd_,LOCK_EX|LOCK_NB)!=0) throw std::runtime_error("Run is already active or cannot be locked");
        auto cfgpath=fs::path(run_dir_)/"run_config.json";
        if(restoring) {
            config_=read_json(cfgpath.string());
            if(config_.at("schema")!=1 || config_.at("program")!=program || config_.at("arguments")!=arguments)
                throw std::runtime_error("Resume program/configuration mismatch");
            if(config_.at("thread_environment")!=thread_environment()) throw std::runtime_error("Thread environment differs from saved run");
            if(config_.contains("executable") && config_.at("executable")!=stamp(arguments.at(0))) throw std::runtime_error("Executable differs from saved run; use original build");
            const auto latest=read_json((fs::path(run_dir_)/"LATEST").string());
            if(latest.at("schema")!=1) throw std::runtime_error("Unknown LATEST schema");
            prune_payloads(run_dir_,latest.at("current"),latest.at("previous"));
        } else {
            if(fs::exists(cfgpath) || fs::exists(fs::path(run_dir_)/"LATEST")) throw std::runtime_error("Run already exists; use --resume_state");
            config_={{"schema",1},{"program",program},{"arguments",arguments},{"working_directory",fs::current_path().string()},
                     {"thread_environment",thread_environment()},{"boundary","completed_optimizer_update"}};
            if(!arguments.empty() && fs::is_regular_file(arguments[0])) config_["executable"]=stamp(arguments[0]);
            write_json_durable(cfgpath.string(),config_); sync_dir(run_dir_);
        }
        mkdir_sync(fs::path(run_dir_)/"training_state"); mkdir_sync(fs::path(run_dir_)/"sessions");
        // The lock excludes a live owner. Recover caches left by an unannounced kill.
        for(const auto& old:fs::directory_iterator(fs::path(run_dir_)/"sessions"))
            if(old.is_directory() && !old.is_symlink()) prune_shard_cache(old.path());
        session_id_=unique_id(); session_dir_=(fs::path(run_dir_)/"sessions"/session_id_).string();
        mkdir_sync(session_dir_); mkdir_sync(fs::path(session_dir_)/"artifacts");
        event_fd_=::open((fs::path(session_dir_)/"events.jsonl").c_str(),O_WRONLY|O_CREAT|O_EXCL|O_CLOEXEC,0600);
        if(event_fd_<0) throw std::system_error(errno,std::generic_category(),"open session events");
        sync_dir(session_dir_);
        // The Android controller clears the previous marker under its launch gate;
        // a marker created after fork belongs to THIS child, even before native init.
        const auto managed = std::getenv("MFT_PARENT_MANAGES_SUSPEND");
        if(restoring && !(managed && std::strcmp(managed, "1") == 0))
            fs::remove(fs::path(run_dir_)/"SUSPEND");
        signal_suspend=0; std::signal(SIGINT,on_signal); std::signal(SIGTERM,on_signal);
        event({{"type","session_start"},{"restoring",restoring},{"pid",::getpid()}});
        if(restoring) readiness_probe_=std::make_unique<PhaseProbe>();
    } catch(...) {
        if(event_fd_>=0) ::close(event_fd_);
        if(lock_fd_>=0) ::close(lock_fd_);
        throw;
    }
}
TrainingSession::~TrainingSession() {
    if(!session_dir_.empty()) prune_shard_cache(session_dir_);
    if(event_fd_>=0) ::close(event_fd_);
    if(lock_fd_>=0) ::close(lock_fd_);
}
void TrainingSession::track_assets(const std::vector<std::string>& files) {
    Json a=Json::array(); std::set<std::string> seen;
    for(const auto& f:files) if(!f.empty()) { auto s=stamp(f); if(seen.insert(s.at("path").get<std::string>()).second) a.push_back(s); }
    std::sort(a.begin(),a.end(),[](const Json& x,const Json& y){return x.at("path")<y.at("path");});
    auto path=fs::path(run_dir_)/"assets.json";
    if(restoring_) { assets_=read_json(path.string()); if(a!=assets_) throw std::runtime_error("Model/data/tokenizer assets changed; resume refused"); }
    else { assets_=a; write_json_durable(path.string(),a); sync_dir(run_dir_); }
    assets_checked_=true;
}
void TrainingSession::request_suspend() { suspend_=true; }
bool TrainingSession::suspend_requested(int updates) const {
    return suspend_.load() || signal_suspend || fs::exists(fs::path(run_dir_)/"SUSPEND") ||
           (suspend_after_>0 && updates>=suspend_after_) ||
           (slice_seconds_>0 && std::chrono::duration<double>(std::chrono::steady_clock::now()-started_).count()>=slice_seconds_);
}
bool TrainingSession::checkpoint_due(int updates) const { return updates%checkpoint_every_==0; }
void TrainingSession::event(Json value) {
    if(!value.is_object()) throw std::runtime_error("Session event must be an object");
    value["session_id"]=session_id_; value["event_seq"]=++event_seq_;
    value["session_elapsed_sec"]=std::chrono::duration<double>(std::chrono::steady_clock::now()-started_).count();
    auto line=value.dump()+"\n"; write_all(event_fd_,line.data(),line.size());
}
std::string TrainingSession::artifact_path(const std::string& name) const {
    check_component(name); return (fs::path(session_dir_)/"artifacts"/name).string();
}
std::vector<std::string> TrainingSession::saved_arguments(const std::string& root,const std::string& program) {
    auto c=read_json((fs::path(root)/"run_config.json").string());
    if(c.at("schema")!=1 || c.at("program")!=program) throw std::runtime_error("Wrong resume entrypoint/schema");
    return c.at("arguments").get<std::vector<std::string>>();
}
std::optional<RestoredState> TrainingSession::restore(const NamedParameters& params,Adam& optimizer,
                                                       const PrepareRestore& prepare) {
    if(!restoring_) return std::nullopt;
    if(!assets_checked_ || restored_) throw std::runtime_error("Invalid restore lifecycle");
    PhaseProbe probe;
    Json latest=read_json((fs::path(run_dir_)/"LATEST").string());
    if(latest.at("schema")!=1) throw std::runtime_error("Unknown LATEST schema");
    std::string selected=latest.at("current"); std::string prev=latest.at("previous");
    TensorMap ts; Json s; bool fallback=false;
    std::function<void()> apply;
    auto load=[&](const std::string& g) {
        auto candidate=validate_generation(run_dir_,g,ts);
        std::set<std::string> names;
        for(const auto& [name,p]:params) {
            check_tensor(p); if(!names.insert(name).second) throw std::runtime_error("Duplicate parameter name");
            auto key="parameter/"+name;
            if(!ts.count(key) || ts.at(key)->shape()!=p->shape()) throw InvalidGeneration("Checkpoint trainable parameter mismatch: "+name);
        }
        size_t count=0;
        for(const auto& [n,t]:ts) {
            if(n.rfind("parameter/",0)==0) ++count;
            else if(n.rfind("optimizer/",0)!=0) throw InvalidGeneration("Unknown checkpoint tensor namespace");
        }
        if(count!=params.size()) throw InvalidGeneration("Unexpected checkpoint parameters");
        AdamSnapshot a; a.metadata=candidate.at("optimizer");
        for(const auto& [n,t]:ts) if(n.rfind("optimizer/",0)==0) a.tensors[n.substr(10)]=t;
        // import_named_state validates the entire candidate before changing optimizer state.
        auto prepared = validate_content([&] {
            return prepare ? prepare(candidate.at("trainer"),candidate.at("dataset_state").get<std::string>())
                           : std::function<void()>{};
        });
        validate_content([&] { optimizer.import_named_state(params,a); });
        apply = std::move(prepared);
        return candidate;
    };
    auto candidate_state=[&](const std::string& g) {
        try { return load(g); }
        catch(const Json::exception& e) { throw InvalidGeneration(e.what()); }
    };
    try { s=candidate_state(selected); }
    catch(const std::exception& e) {
        if(prev.empty() || prev==selected) throw;
        std::cerr << "Current checkpoint invalid (" << e.what() << "); trying recorded predecessor only\n";
        selected=prev;
        try { s=candidate_state(selected); }
        catch(const InvalidGeneration& prior) {
            if(dynamic_cast<const InvalidGeneration*>(&e))
                throw CheckpointsExhausted(std::string("Both committed checkpoints are invalid: ")+e.what()+"; "+prior.what());
            throw;
        }
        fallback=true;
    }
    if (apply) apply();
    for(const auto& [name,p]:params) std::memcpy(p->data<float>(),ts.at("parameter/"+name)->data<float>(),static_cast<size_t>(p->numel())*sizeof(float));
    current_=selected; previous_=s.at("manifest").at("previous").get<std::string>(); restored_=true;
    event({{"type","resume"},{"generation",selected},{"fallback",fallback},
           {"completed_updates",s.at("trainer").at("completed_updates")},{"state_restore_metrics",probe.finish()}});
    return RestoredState{s.at("trainer"),s.at("dataset_state"),selected,s.at("status"),fallback};
}
void TrainingSession::resume_ready() {
    if(!restored_ || !readiness_probe_) throw std::runtime_error("Invalid resume-ready boundary");
    event({{"type","resume_ready"},{"metrics",readiness_probe_->finish()}});
    readiness_probe_.reset();
}
void TrainingSession::commit(const NamedParameters& params,const Adam& optimizer,
    const std::string& dataset,const Json& trainer,const std::string& status) {
    if(!assets_checked_ || poisoned_ || (restoring_&&!restored_)) throw std::runtime_error("Invalid commit lifecycle");
    if(status!="running" && status!="suspended" && status!="completed") throw std::runtime_error("Invalid checkpoint status");
    check_progress(trainer);
    if(!Json::parse(dataset).is_object()) throw std::runtime_error("Invalid dataset state");
    auto hook=[&](const std::string& stage){if(test_hook_) test_hook_(stage);};
    const auto start=std::chrono::steady_clock::now(); bool pointer_replaced=false;
    PhaseProbe probe;
    try {
        AdamSnapshot a=optimizer.export_named_state(params); TensorMap tensors;
        for(const auto& [name,p]:params) if(!tensors.emplace("parameter/"+name,p).second) throw std::runtime_error("Duplicate named parameter");
        for(const auto& [name,p]:a.tensors) tensors.emplace("optimizer/"+name,p);
        std::string gen="g-"+session_id_+"-"+std::to_string(++generation_counter_);
        fs::path parent=fs::path(run_dir_)/"training_state", tmp=parent/(gen+".tmp"), dest=parent/gen;
        mkdir_sync(tmp);
        write_tensor_file((tmp/"state.safetensors").string(),tensors); hook("after_tensor_write");
        Json state={{"schema",1},{"trainer",trainer},{"optimizer",a.metadata},{"dataset_state",dataset},{"status",status}};
        write_json_durable((tmp/"state.json").string(),state); hook("after_state_write");
        Json best=nullptr;
        if(trainer.contains("best_path") && !trainer.at("best_path").is_null() && !trainer.at("best_path").get<std::string>().empty()) {
            std::string p=trainer.at("best_path"); Fd fd(::open(p.c_str(),O_RDONLY|O_CLOEXEC)); sync_fd(fd.value); sync_dir(fs::path(p).parent_path()); best=stamp(p);
        }
        sync_fd(event_fd_);
        Json manifest={{"schema",1},{"generation",gen},{"previous",current_},{"session_id",session_id_},{"event_seq",event_seq_},
            {"completed_updates",trainer.at("completed_updates")},{"best_file",best},
            {"files",{{"state.json",fs::file_size(tmp/"state.json")},{"state.safetensors",fs::file_size(tmp/"state.safetensors")}}}};
        write_json_durable((tmp/"manifest.json").string(),manifest); sync_dir(tmp);
        hook("before_generation_rename"); fs::rename(tmp,dest); sync_dir(parent); hook("after_generation_rename");
        fs::path pointer_tmp=fs::path(run_dir_)/("LATEST."+session_id_+".tmp");
        write_json_durable(pointer_tmp.string(),{{"schema",1},{"current",gen},{"previous",current_}});
        hook("before_pointer_rename"); fs::rename(pointer_tmp,fs::path(run_dir_)/"LATEST"); pointer_replaced=true;
        hook("after_pointer_rename"); sync_dir(run_dir_);
        previous_=current_; current_=gen;
        event({{"type","checkpoint_committed"},{"generation",gen},{"completed_updates",trainer.at("completed_updates")},{"status",status},
               {"save_time_sec",std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()},
               {"save_metrics",probe.finish()},
               {"payload_bytes",manifest["files"]["state.safetensors"]}});
        sync_fd(event_fd_);
        // Keep the complete current and previous states. Small manifests retain the full log lineage.
        prune_payloads(run_dir_,current_,previous_);
    } catch(...) { if(pointer_replaced) poisoned_=true; throw; }
}
Invocation prepare_invocation(int argc,char** argv,const std::string& program,
                              const BatchValidator& validate_batch) {
    Invocation inv; inv.arguments.emplace_back(argv[0]); std::string root; bool resume=false,has_run=false;
    int every=1,stop=0; double seconds=0; bool controls=false;
    for(int i=1;i<argc;++i) {
        std::string key=argv[i];
        if(key=="--run_dir" || key=="--resume_state" || key=="--checkpoint_every_updates" || key=="--suspend_after_updates" || key=="--slice_seconds" || key=="--resume_microbatch") {
            controls=true; if(i+1>=argc) throw std::runtime_error("Missing value for "+key); std::string value=argv[++i]; size_t pos=0;
            if(key=="--run_dir" || key=="--resume_state") { if(has_run) throw std::runtime_error("Choose one run or resume directory"); root=value; has_run=true; resume=key=="--resume_state"; }
            else if(key=="--resume_microbatch") {
                if(inv.resume_microbatch || value.empty() || value.find_first_not_of("0123456789")!=std::string::npos)
                    throw std::runtime_error("Specify --resume_microbatch once, as a positive integer");
                inv.resume_microbatch=std::stoi(value);
                if(*inv.resume_microbatch<1) throw std::runtime_error("--resume_microbatch must be positive");
            }
            else if(key=="--slice_seconds") { seconds=std::stod(value,&pos); if(pos!=value.size()) throw std::runtime_error("Invalid slice_seconds"); }
            else { int n=std::stoi(value,&pos); if(pos!=value.size()) throw std::runtime_error("Invalid update control"); if(key=="--checkpoint_every_updates") every=n; else stop=n; }
        } else inv.arguments.push_back(key);
    }
    if(inv.resume_microbatch && !resume) throw std::runtime_error("--resume_microbatch requires --resume_state");
    if(!has_run) { if(controls) throw std::runtime_error("Resume controls require --run_dir or --resume_state"); return inv; }
#ifdef __ANDROID__
    const auto parent=::getppid();
    if(::prctl(PR_SET_PDEATHSIG,SIGKILL)!=0 || ::getppid()!=parent || parent==1)
        throw std::runtime_error("Cannot bind trainer lifetime to Android controller");
#endif
    const auto executable = running_executable();
    inv.arguments[0] = executable;
    root=fs::absolute(root).lexically_normal().string();
    if(resume) {
        if(inv.arguments.size()!=1) throw std::runtime_error("Resume restores original training arguments; do not override them");
        auto cfg=read_json((fs::path(root)/"run_config.json").string());
        auto args=TrainingSession::saved_arguments(root,program);
        // Validate the actual binary, not just the saved path to it.
        if(!cfg.contains("executable") || cfg.at("executable")!=stamp(executable)) throw std::runtime_error("Run resume using the original executable");
        fs::current_path(cfg.at("working_directory").get<std::string>()); inv.arguments=std::move(args);
        if(inv.resume_microbatch) validate_batch(inv.arguments,*inv.resume_microbatch);
    }
    inv.session=std::make_unique<TrainingSession>(root,resume,program,inv.arguments,every,stop,seconds);
    return inv;
}
}  // namespace ops::resume
