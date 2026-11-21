#pragma once

#include "../optim/adam.h"
#include "phase_probe.h"
#include "../../third_party/nlohmann/json.hpp"
#include <atomic>
#include <chrono>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace ops::resume {
using Json = nlohmann::json;
using TensorMap = std::map<std::string, TensorPtr>;

// Process boundary contract shared with the Android TrainingRunner.
struct CheckpointsExhausted : std::runtime_error {
    using std::runtime_error::runtime_error;
    static constexpr int exit_code = 65;
};

struct RestoredState {
    Json trainer;
    std::string dataset_state;
    std::string generation;
    std::string status;
    bool fallback = false;
};

// A session is an update-boundary transaction, not a saved autograd graph.
class TrainingSession {
public:
    TrainingSession(const std::string& run_dir, bool restoring,
                    const std::string& program, const std::vector<std::string>& arguments,
                    int checkpoint_every, int suspend_after, double slice_seconds);
    ~TrainingSession();
    TrainingSession(const TrainingSession&) = delete;
    TrainingSession& operator=(const TrainingSession&) = delete;

    const std::string& run_dir() const { return run_dir_; }
    const std::string& session_dir() const { return session_dir_; }
    const std::string& session_id() const { return session_id_; }
    bool restoring() const { return restoring_; }
    void track_assets(const std::vector<std::string>& files);
    void request_suspend();
    bool suspend_requested(int completed_updates) const;
    bool checkpoint_due(int completed_updates) const;
    void event(Json value);
    void resume_ready();
    using PrepareRestore = std::function<std::function<void()>(const Json&, const std::string&)>;
    std::optional<RestoredState> restore(const NamedParameters& parameters, Adam& optimizer,
                                         const PrepareRestore& prepare = {});
    void commit(const NamedParameters& parameters, const Adam& optimizer,
                const std::string& dataset_state, const Json& trainer,
                const std::string& status);
    std::string artifact_path(const std::string& name) const;
    // Fault injection for durability tests; normal callers leave this unset.
    void set_commit_test_hook(std::function<void(const std::string&)> hook) { test_hook_ = std::move(hook); }
    static std::vector<std::string> saved_arguments(const std::string& run_dir,
                                                  const std::string& program);

private:
    std::string run_dir_, session_dir_, session_id_, program_, current_, previous_;
    int lock_fd_ = -1, event_fd_ = -1;
    bool restoring_, assets_checked_ = false, restored_ = false, poisoned_ = false;
    int checkpoint_every_, suspend_after_;
    double slice_seconds_;
    uint64_t event_seq_ = 0, generation_counter_ = 0;
    std::atomic<bool> suspend_{false};
    std::chrono::steady_clock::time_point started_;
    Json config_, assets_;
    std::function<void(const std::string&)> test_hook_;
    std::unique_ptr<PhaseProbe> readiness_probe_;
};

struct Invocation {
    std::vector<std::string> arguments;
    std::unique_ptr<TrainingSession> session;
    std::optional<int> resume_microbatch;
};

// Consumes only --run_dir/--resume_state/--checkpoint_every_updates/
// --suspend_after_updates/--slice_seconds/--resume_microbatch. Training arguments
// remain immutable. Validate a physical override before opening a writable session.
using BatchValidator = std::function<void(std::vector<std::string>&, int)>;
Invocation prepare_invocation(int argc, char** argv, const std::string& program,
                              const BatchValidator& validate_batch);
void write_tensor_file(const std::string& path, const TensorMap& tensors);
TensorMap read_tensor_file(const std::string& path);
Json read_json(const std::string& path);
void write_json_durable(const std::string& path, const Json& value);
}  // namespace ops::resume
