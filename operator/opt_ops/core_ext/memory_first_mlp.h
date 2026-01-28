/**
 * @file memory_first_mlp.h
 * [Documentation available in English]
 * 
 * corestrategy：
 * [Documentation available in English]
 *    requirecompletestorage hidden [batch*seq, n_inner]（for example 2*64*3072*4B = 1.5MB）
 * 
 * [Documentation available in English]
 *    - Compute hidden_chunk_i = GELU(x @ W1[:,i_start:i_end] + b1[i_start:i_end])
 * [Documentation available in English]
 *    - release hidden_chunk_i
 * [Documentation available in English]
 * 
 * [Documentation available in English]
 */

#pragma once

#include "tensor.h"
#include "ops.h"

namespace ops {
namespace memory_first {

/**
 * [Documentation available in English]
 * 
 * @param input input [batch*seq_len, n_embd]
 * @param fc_weight first layerweight [n_inner, n_embd]（transposeback matmul）
 * @param fc_bias first layerbias [n_inner]
 * @param proj_weight second layerweight [n_embd, n_inner]（transposeback matmul）
 * @param proj_bias second layerbias [n_embd]
 * @param chunk_size intermediatelayerchunkedsize（default 256）
 * @return output [batch*seq_len, n_embd]
 * 
 * [Documentation available in English]
 * 1. output = zeros([batch*seq_len, n_embd])
 * 2. for chunk_start in range(0, n_inner, chunk_size):
 *      chunk_end = min(chunk_start + chunk_size, n_inner)
 *      # Computeintermediatelayeroneblock
 *      hidden_chunk = input @ fc_weight[:, chunk_start:chunk_end].T + fc_bias[chunk_start:chunk_end]
 *      activated_chunk = GELU(hidden_chunk)
 * [Documentation available in English]
 *      output += activated_chunk @ proj_weight[:, chunk_start:chunk_end].T
 * [Documentation available in English]
 * 3. output += proj_bias
 * 4. return output
 */
TensorPtr memory_first_mlp_forward(
    const TensorPtr& input,
    const TensorPtr& fc_weight,
    const TensorPtr& fc_bias,
    const TensorPtr& proj_weight,
    const TensorPtr& proj_bias,
    int chunk_size = 256
);

} // namespace memory_first
} // namespace ops

