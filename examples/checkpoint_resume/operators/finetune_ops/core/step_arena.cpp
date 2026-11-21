/**
 * @file step_arena.cpp
 * @brief Implementation of step-based memory arena allocator
 */

#include "step_arena.h"

namespace ops {

StepArena& get_step_arena() {
    // 16MB arena (for mobile and resource-constrained environments)
    static StepArena arena(16);
    return arena;
}

} // namespace ops

