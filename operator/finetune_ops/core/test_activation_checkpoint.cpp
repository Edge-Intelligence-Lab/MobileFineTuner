#include "checkpoint.h"
#include "autograd_engine.h"
#include "ops.h"

#include <cmath>
#include <iostream>
#include <stdexcept>

using namespace ops;

namespace {
void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
#ifdef USE_NEW_AUTOGRAD_ENGINE
void close(float actual, float expected, float tolerance = 2e-5f) {
    if (!std::isfinite(actual) || std::fabs(actual - expected) > tolerance)
        throw std::runtime_error("gradient/value mismatch: " + std::to_string(actual) +
                                 " vs " + std::to_string(expected));
}
#endif
template<class Fn> void throws(Fn fn) {
    bool failed = false;
    try { fn(); } catch (const std::exception&) { failed = true; }
    check(failed, "expected exception");
}
#ifdef USE_NEW_AUTOGRAD_ENGINE
TensorPtr variable(std::initializer_list<float> values) {
    auto result = tensor(values);
    result->set_requires_grad(true);
    return result;
}

TensorPtr graph(const TensorPtr& x, const TensorPtr& p, bool checkpointed,
                bool nested, bool reverse, int& calls) {
    auto captured = mul(p, p); // Non-leaf capture must propagate to p, not stop here.
    auto input = mul(x, 2.0f); // Non-leaf positional input as well.
    auto fn = [&](const TensorPtr& value) {
        ++calls;
        return add(mul(value, captured), mul(p, p));
    };
    // Capture by value: checkpoint outlives this stack frame.
    auto block = [captured, p, &calls, nested](const TensorPtr& value) {
        ++calls;
        auto inner = [captured, p](const TensorPtr& t) { return add(mul(t, captured), mul(p, p)); };
        return nested ? checkpoint(inner, value, {p, captured}) : inner(value);
    };
    TensorPtr direct;
    if (!reverse) direct = add(mul(x, p), mul(captured, 3.0f));
    auto y = checkpointed ? checkpoint(block, input, {p, captured, p}) : fn(input);
    if (reverse) direct = add(mul(x, p), mul(captured, 3.0f));
    return sum(reverse ? add(y, direct) : add(direct, y), -1, false);
}

void parity_and_fd(bool nested, bool reverse) {
    auto x = variable({0.3f, -0.7f, 1.2f});
    auto p = variable({-0.4f, 0.6f, 0.9f});
    auto xc = x->clone(), pc = p->clone();
    int ordinary_calls = 0, checkpoint_calls = 0;
    for (int step = 0; step < 2; ++step) {
        x->zero_grad(); p->zero_grad(); xc->zero_grad(); pc->zero_grad();
        auto loss = graph(x, p, false, false, reverse, ordinary_calls);
        const float reference = loss->item();
        loss->backward();
        auto saved_x = x->grad()->clone(), saved_p = p->grad()->clone();
        auto checkpoint_loss = graph(xc, pc, true, nested, reverse, checkpoint_calls);
        close(checkpoint_loss->item(), reference);
        checkpoint_loss->backward();
        for (int i = 0; i < 3; ++i) {
            close(xc->grad()->data<float>()[i], saved_x->data<float>()[i]);
            close(pc->grad()->data<float>()[i], saved_p->data<float>()[i]);
            for (auto target : {x, p}) {
                auto& value = target->data<float>()[i];
                const float original = value, epsilon = 1e-3f;
                auto& engine = autograd::Engine::instance();
                engine.set_enabled(false);
                value = original + epsilon;
                float plus = graph(x, p, false, false, reverse, ordinary_calls)->item();
                value = original - epsilon;
                float minus = graph(x, p, false, false, reverse, ordinary_calls)->item();
                value = original;
                engine.set_enabled(true);
                close(target->grad()->data<float>()[i], (plus - minus) / (2 * epsilon), 1e-3f);
            }
        }
        // Real parameter updates; rebuild both graphs on the next step.
        for (int i = 0; i < 3; ++i) {
            p->data<float>()[i] -= 0.03f * p->grad()->data<float>()[i];
            pc->data<float>()[i] -= 0.03f * pc->grad()->data<float>()[i];
        }
    }
    check(checkpoint_calls == 4, "checkpoint must replay once per step");
}

void boundaries_and_accumulation() {
    auto x = variable({2.0f}), p = variable({3.0f});
    auto frozen = tensor({4.0f});
    p->set_grad(tensor({7.0f}));
    auto y = checkpoint([p](const TensorPtr& t) { return mul(t, p); }, frozen, {p});
    y->backward();
    close(p->grad()->item(), 11.0f); // No grad-enabled positional input needed.
    auto identity = checkpoint([](const TensorPtr& t) { return t; }, x);
    check(identity != x, "identity checkpoint must not create a cycle");
    add(identity, mul(x, 3.0f))->backward();
    close(x->grad()->item(), 4.0f);
    x->zero_grad();
    auto both = checkpoint([x](const TensorPtr& t) { return add(t, x); }, x, {x});
    both->backward();
    close(x->grad()->item(), 2.0f);
    p->zero_grad();
    auto returned = checkpoint([p](const TensorPtr&) { return p; }, frozen, {p});
    returned->backward();
    close(p->grad()->item(), 1.0f);
    x->zero_grad();
    auto a = checkpoint([p](const TensorPtr& t) { return mul(t, p); }, x, {p});
    auto b = checkpoint([p](const TensorPtr& t) { return mul(t, p); }, x, {p});
    add(a, b)->backward();
    close(x->grad()->item(), 6.0f);
    close(p->grad()->item(), 5.0f);
}

void exception_restoration() {
    auto& engine = autograd::Engine::instance();
    auto x = variable({2.0f}), p = variable({3.0f});
    auto direct = mul(x, p);
    auto original_node = engine.get_node(direct);
    throws([&] { checkpoint([](const TensorPtr&) -> TensorPtr {
        throw std::runtime_error("forward failure");
    }, x); });
    check(engine.is_enabled(), "forward did not restore enabled");
    throws([&] { checkpoint([](const TensorPtr&) { return TensorPtr{}; }, x); });
    int calls = 0;
    auto y = checkpoint([&](const TensorPtr& t) {
        auto intermediate = mul(t, p);
        if (++calls == 2) throw std::runtime_error("recompute failure");
        return intermediate;
    }, x, {p});
    auto checkpoint_node = engine.get_node(y);
    throws([&] { checkpoint_node->backward_fn->apply(ones({1})); });
    check(engine.is_enabled(), "recompute did not restore enabled");
    check(engine.get_node(direct) == original_node, "outer graph was replaced");
    check(!p->grad(), "failed recompute leaked parameter gradients");
    direct->backward();
    close(x->grad()->item(), 3.0f); close(p->grad()->item(), 2.0f);
    class FailingBackward : public BackwardFunction {
    public:
        std::vector<TensorPtr> apply(const TensorPtr&) override {
            throw std::runtime_error("nested backward failure");
        }
    };
    auto preserved = mul(x, p);
    auto preserved_node = engine.get_node(preserved);
    auto failing = checkpoint([&](const TensorPtr& t) {
        auto intermediate = mul(t, p);
        auto result = intermediate->clone();
        engine.register_node(result, {intermediate}, std::make_shared<FailingBackward>());
        return result;
    }, x, {p});
    auto failing_node = engine.get_node(failing);
    engine.set_enabled(false);
    throws([&] { failing_node->backward_fn->apply(ones({1})); });
    check(!engine.is_enabled(), "nested backward did not restore disabled state");
    check(engine.get_node(preserved) == preserved_node, "nested backward lost outer node");
    close(p->grad()->item(), 2.0f);
    engine.set_enabled(true);
    preserved->backward();
    close(x->grad()->item(), 6.0f); close(p->grad()->item(), 4.0f);
    engine.set_enabled(false);
    throws([&] { checkpoint([](const TensorPtr&) -> TensorPtr {
        throw std::runtime_error("disabled failure");
    }, x); });
    check(!engine.is_enabled(), "disabled forward changed engine state");
    engine.set_enabled(true);
    throws([&] { checkpoint({}, x); });
    throws([&] { checkpoint([](const TensorPtr& t) { return t; }, nullptr); });
    throws([&] { checkpoint([](const TensorPtr& t) { return t; }, empty({1}, kFloat16)); });
    engine.clear_graph();
}

void activation_lifetime() {
    auto x = variable({2.0f});
    std::weak_ptr<Tensor> intermediate;
    auto y = checkpoint([&](const TensorPtr& t) {
        auto temporary = mul(t, t);
        intermediate = temporary;
        return mul(temporary, 3.0f);
    }, x);
    check(intermediate.expired(), "checkpoint retained forward activation");
    y->backward();
    close(x->grad()->item(), 12.0f);
    check(intermediate.expired(), "checkpoint retained replay activation");
}
#endif
} // namespace

int main() {
    try {
#ifdef USE_NEW_AUTOGRAD_ENGINE
        for (bool nested : {false, true}) for (bool reverse : {false, true})
            parity_and_fd(nested, reverse);
        boundaries_and_accumulation();
        exception_restoration();
        activation_lifetime();
#else
        throws([] { checkpoint([](const TensorPtr& t) { return t; }, tensor({1.0f})); });
#endif
        std::cout << "[PASS] activation checkpoint primitives\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "[FAIL] " << error.what() << '\n';
        return 1;
    }
}
