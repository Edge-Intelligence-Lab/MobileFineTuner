#include "checkpoint.h"
#include "autograd_engine.h"

#include <algorithm>

namespace ops {
namespace {

void validate(const TensorPtr& tensor) {
    if (!tensor || tensor->dtype() != kFloat32 || !tensor->device().is_cpu()) {
        throw TensorError("checkpoint requires non-null CPU FP32 tensors");
    }
}

class NestedState {
public:
    explicit NestedState(autograd::Engine& engine)
        : engine_(engine), enabled_(engine.is_enabled()), state_(engine.take_state()) {
        engine_.set_enabled(true);
    }
    ~NestedState() {
        engine_.restore_state(std::move(state_));
        engine_.set_enabled(enabled_);
    }
    NestedState(const NestedState&) = delete;
    NestedState& operator=(const NestedState&) = delete;
private:
    autograd::Engine& engine_;
    bool enabled_;
    autograd::Engine::EngineState state_;
};

class CheckpointBackward : public BackwardFunction {
public:
    CheckpointBackward(std::function<TensorPtr(const TensorPtr&)> fn, TensorPtr input,
                       std::vector<TensorPtr> dependencies, std::vector<int64_t> shape)
        : fn_(std::move(fn)), input_(std::move(input)),
          dependencies_(std::move(dependencies)), shape_(std::move(shape)) {}

    std::vector<TensorPtr> apply(const TensorPtr& grad_output) override {
        validate(grad_output);
        if (grad_output->shape() != shape_) throw TensorError("checkpoint gradient shape mismatch");
        auto replay_input = input_->clone();
        auto& engine = autograd::Engine::instance();
        NestedState scope(engine);
        auto output = fn_(replay_input);
        validate(output);
        if (output->shape() != shape_) throw TensorError("checkpoint replay shape changed");
        auto gradients = engine.backward_gradients(output, grad_output);
        std::vector<TensorPtr> result;
        for (const auto& dependency : dependencies_) {
            TensorPtr gradient;
            auto found = gradients.find(dependency.get());
            if (found != gradients.end()) gradient = found->second->clone();
            // The closure can also capture the original input directly.
            if (dependency == input_) {
                auto replay = gradients.find(replay_input.get());
                if (replay != gradients.end()) {
                    if (!gradient) gradient = replay->second->clone();
                    else for (int64_t i = 0; i < gradient->numel(); ++i)
                        gradient->data<float>()[i] += replay->second->data<float>()[i];
                }
            }
            result.push_back(gradient);
        }
        return result;
    }
private:
    std::function<TensorPtr(const TensorPtr&)> fn_;
    TensorPtr input_;
    std::vector<TensorPtr> dependencies_;
    std::vector<int64_t> shape_;
};

} // namespace

TensorPtr checkpoint(const std::function<TensorPtr(const TensorPtr&)>& fn,
                     const TensorPtr& input,
                     const std::vector<TensorPtr>& captured_tensors) {
#ifndef USE_NEW_AUTOGRAD_ENGINE
    (void)fn;
    (void)input;
    (void)captured_tensors;
    throw TensorError("checkpoint requires USE_NEW_AUTOGRAD_ENGINE");
#else
    if (!fn) throw TensorError("checkpoint requires a callable");
    validate(input);
    for (const auto& capture : captured_tensors) {
        validate(capture);
        if (!capture->requires_grad()) throw TensorError("checkpoint explicit captures must require gradients");
    }
    auto& engine = autograd::Engine::instance();
    // Nested no-grad forward is part of the enclosing capture, not a new node.
    if (!engine.is_enabled()) {
        auto output = fn(input);
        validate(output);
        return output;
    }
    std::vector<TensorPtr> dependencies;
    auto output = engine.capture_dependencies([&] { return fn(input); }, dependencies);
    validate(output);
    for (const auto& capture : captured_tensors)
        if (std::find(dependencies.begin(), dependencies.end(), capture) == dependencies.end())
            dependencies.push_back(capture);
    for (const auto& dependency : dependencies) validate(dependency);
    if (dependencies.empty()) return output;
    // Avoid a self-edge when fn returns an input/capture without allocating.
    if (std::find(dependencies.begin(), dependencies.end(), output) != dependencies.end())
        output = output->clone();
    // Some public unary ops install a legacy closure even in new-engine no-grad
    // mode. The checkpoint node replaces it; keeping it would retain activations.
    output->set_grad_fn(nullptr);
    output->set_requires_grad(true);
    engine.register_node(output, dependencies,
        std::make_shared<CheckpointBackward>(fn, input, dependencies, output->shape()));
    return output;
#endif
}

} // namespace ops
