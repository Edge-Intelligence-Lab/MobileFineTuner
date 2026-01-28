/**
 * @file step_arena.cpp
 * [Documentation available in English]
 */

#include "step_arena.h"

namespace ops {

StepArena& get_step_arena() {
        // [Translated]
    // [Translated comment removed - see documentation]
    static StepArena arena(16);  // 16MB arena（mobile）
    return arena;
}

} // namespace ops

