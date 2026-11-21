/**
 * @file checkpoint.h
 * @brief Activation checkpointing utilities (recompute forward during backward)
 *
 * This module provides a lightweight checkpoint wrapper that:
 * - runs forward with autograd disabled (no activation saving)
 * - attaches a custom backward that recomputes forward and runs backward
 *
 * This is critical for reducing peak RSS during training on mobile devices.
 */

#pragma once

#include "tensor.h"
#include <functional>
#include <cstdint>

namespace ops {

struct CheckpointProfileEvent {
    uint64_t call_id = 0;
    const char* phase = "";
    uint64_t input_bytes = 0;
    uint64_t copy_payload_bytes = 0;
    double wall_sec = 0;
    bool success = false;
    bool os_io_available = false;
    uint64_t os_read_bytes = 0;
    uint64_t os_write_bytes = 0;
};

// Thread-local, disabled by default. Payload bytes are not hardware DRAM traffic.
void set_checkpoint_profiling(bool enabled);
std::vector<CheckpointProfileEvent> take_checkpoint_profile_events();

/**
 * @brief Checkpoint a function of a single tensor input
 *
 * @param fn Forward function (takes input tensor, returns output tensor)
 * @param input Input tensor
 * @return Output tensor with custom backward that recomputes forward
 */
TensorPtr checkpoint(const std::function<TensorPtr(const TensorPtr&)>& fn,
                     const TensorPtr& input);

}  // namespace ops
