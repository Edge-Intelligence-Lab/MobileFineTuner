/**
 * @file checkpoint.cpp
 * @brief Activation checkpointing implementation
 */

#include "checkpoint.h"
#include "autograd_engine.h"
#include "backward_functions.h"
#include "ops.h"

#include <iostream>
#include <chrono>
#ifdef __APPLE__
#include <libproc.h>
#include <unistd.h>
#endif

namespace ops {

namespace {

thread_local bool profile_enabled = false;
thread_local uint64_t next_profile_id = 0;
thread_local std::vector<CheckpointProfileEvent> profile_events;

struct ProcessIO { bool available = false; uint64_t read = 0, written = 0; };
ProcessIO process_io() {
    ProcessIO out;
#ifdef __APPLE__
    rusage_info_v2 info{};
    if (proc_pid_rusage(getpid(), RUSAGE_INFO_V2, reinterpret_cast<rusage_info_t*>(&info)) == 0) {
        out = {true, info.ri_diskio_bytesread, info.ri_diskio_byteswritten};
    }
#endif
    return out;
}

template <typename Fn>
TensorPtr profile_phase(uint64_t id, const char* phase, const TensorPtr& input,
                        bool copies_input, Fn&& fn) {
    if (!profile_enabled) return fn();
    const auto before = process_io();
    const auto started = std::chrono::steady_clock::now();
    auto record = [&](bool success) {
        const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
        const auto after = process_io();
        const bool available = before.available && after.available &&
                               after.read >= before.read && after.written >= before.written;
        const uint64_t bytes = static_cast<uint64_t>(input->numel()) * DTypeUtils::size_of(input->dtype());
        profile_events.push_back({id, phase, bytes, copies_input && success ? bytes : 0,
            elapsed, success, available, available ? after.read-before.read : 0,
            available ? after.written-before.written : 0});
    };
    TensorPtr result;
    try {
        result = fn();
    } catch (...) {
        record(false);
        throw;
    }
    record(true);
    return result;
}

bool ckpt_debug_enabled() {
    const char* v = std::getenv("OPS_CKPT_DEBUG");
    if (!v) return false;
    return std::string(v) == "1";
}

class CheckpointBackward : public BackwardFunction {
public:
    CheckpointBackward(std::function<TensorPtr(const TensorPtr&)> fn,
                       TensorPtr input, uint64_t profile_id)
        : fn_(std::move(fn)), input_(std::move(input)), profile_id_(profile_id) {}

    std::vector<TensorPtr> apply(const TensorPtr& grad_output) override {
        if (!input_) {
            throw TensorError("CheckpointBackward: input is null");
        }

        if (ckpt_debug_enabled()) {
            std::cout << "[CKPT] backward begin, input=" << input_.get()
                      << " shape=" << input_->to_string() << std::endl;
        }

        // Create a safe detached copy for recompute to avoid aliasing issues
        auto input_view = profile_phase(profile_id_, "clone_input", input_, true,
                                       [&] { return input_->clone(); });
        input_view->set_requires_grad(true);

        auto& eng = autograd::Engine::instance();
        const bool prev_enabled = eng.is_enabled();
        const bool prev_suppress = eng.suppress_external_grad_nodes();

        // Swap out engine state to allow nested backward safely
        auto saved_state = eng.take_state();
        eng.set_enabled(true);
        eng.set_suppress_external_grad_nodes(true);

        // Recompute forward with autograd enabled
        if (ckpt_debug_enabled()) {
            std::cout << "[CKPT] recompute forward..." << std::endl;
        }
        try {
            auto output = profile_phase(profile_id_, "recompute_forward", input_, false,
                                        [&] { return fn_(input_view); });
            if (!output) throw TensorError("CheckpointBackward: recompute returned null");
            if (!output->requires_grad()) output->set_requires_grad(true);
            profile_phase(profile_id_, "nested_backward", input_, false, [&] {
                output->backward(grad_output);
                return TensorPtr{};
            });
        } catch (...) {
            eng.clear_graph();
            eng.set_enabled(prev_enabled);
            eng.set_suppress_external_grad_nodes(prev_suppress);
            eng.restore_state(std::move(saved_state));
            throw;
        }

        // Restore engine state
        eng.set_enabled(prev_enabled);
        eng.set_suppress_external_grad_nodes(prev_suppress);
        eng.restore_state(std::move(saved_state));

        auto grad_input = input_view->grad();
        if (!grad_input) {
            grad_input = zeros(input_->shape(), input_->dtype(), input_->device());
        }

        if (ckpt_debug_enabled()) {
            std::cout << "[CKPT] backward done, grad_input=" << grad_input.get() << std::endl;
        }

        return {grad_input};
    }

private:
    std::function<TensorPtr(const TensorPtr&)> fn_;
    TensorPtr input_;
    uint64_t profile_id_;
};

}  // namespace

void set_checkpoint_profiling(bool enabled) { profile_enabled = enabled; }

std::vector<CheckpointProfileEvent> take_checkpoint_profile_events() {
    std::vector<CheckpointProfileEvent> out;
    out.swap(profile_events);
    return out;
}

TensorPtr checkpoint(const std::function<TensorPtr(const TensorPtr&)>& fn,
                     const TensorPtr& input) {
    if (!fn || !input) {
        throw TensorError("checkpoint: fn or input is null");
    }

    auto& eng = autograd::Engine::instance();
    const bool prev_enabled = eng.is_enabled();

    // Forward with autograd disabled to avoid saving activations
    if (ckpt_debug_enabled()) {
        std::cout << "[CKPT] forward begin, input=" << input.get()
                  << " shape=" << input->to_string() << std::endl;
    }
    eng.set_enabled(false);
    TensorPtr output;
    try {
        output = fn(input);
        if (!output) throw TensorError("checkpoint: forward returned null");
    } catch (...) {
        eng.set_enabled(prev_enabled);
        throw;
    }
    eng.set_enabled(prev_enabled);

    // Attach custom backward if needed
    if (input->requires_grad()) {
        output->set_requires_grad(true);
        const auto id = profile_enabled ? ++next_profile_id : 0;
        profile_phase(id, "retain_input", input, false, [&] {
            auto backward_fn = std::make_shared<CheckpointBackward>(fn, input, id);
            eng.register_node(output, {input}, backward_fn);
            return TensorPtr{};
        });
    }

    if (ckpt_debug_enabled()) {
        std::cout << "[CKPT] forward end, output=" << output.get()
                  << " shape=" << output->to_string() << std::endl;
    }

    return output;
}

}  // namespace ops
