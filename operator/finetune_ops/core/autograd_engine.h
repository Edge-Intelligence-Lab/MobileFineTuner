/**
 * @file autograd_engine.h
 * @brief Topological-sort based autograd engine for efficient backward pass
 * 
 * This engine replaces recursive backward with iterative topological sorting,
 * eliminating deep call stacks and enabling training of deep networks (6-12+ layers)
 * on CPU with reasonable memory and time.
 */

#pragma once

#include "tensor.h"
#include "backward_functions.h"
#include <vector>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <queue>

namespace ops {
namespace autograd {

// Forward declarations
class Node;
class Edge;

using NodePtr = std::shared_ptr<Node>;
using EdgePtr = std::shared_ptr<Edge>;

/**
 * @brief Edge connecting nodes in the computation graph
 * 
 * Each edge represents a dependency: output tensor depends on input tensor
 * through a specific backward function.
 */
class Edge {
public:
    NodePtr input_node;   // Source node (input)
    int input_idx;        // Index of this input in the backward function
    
    Edge(NodePtr input, int idx)
        : input_node(input), input_idx(idx) {}
};

/**
 * @brief Node in the computation graph representing a tensor operation
 * 
 * Each node stores:
 * - The output tensor
 * - Backward function to compute gradients
 * - Edges to input nodes
 * - Reference count for memory management
 */
class Node {
public:
    TensorPtr tensor;                           // The tensor this node represents
    BackwardFunctionPtr backward_fn;            // How to compute gradients
    std::vector<EdgePtr> next_edges;            // Edges to input nodes
    int ref_count;                              // Number of outputs depending on this node
    bool ready;                                 // Whether all consumers have been processed
    
    Node(const TensorPtr& t) : tensor(t), ref_count(0), ready(false) {}
    
    void add_next_edge(const EdgePtr& edge) {
        next_edges.push_back(edge);
    }
    
    void set_backward_fn(BackwardFunctionPtr fn) {
        backward_fn = fn;
    }
};

/**
 * @brief Main autograd engine that executes backward pass iteratively
 * 
 * Features:
 * - Topological sorting of computation graph
 * - Iterative (non-recursive) gradient computation
 * - Automatic memory cleanup of intermediate tensors
 * - Support for multiple outputs and gradient accumulation
 */
class Engine {
public:
    static Engine& instance() {
        static Engine engine;
        return engine;
    }
    
    /**
     * @brief Execute backward pass from given outputs
     * 
     * @param outputs Tensors to compute gradients from (e.g., loss)
     * @param output_grads Initial gradients for outputs (defaults to ones)
     */
    void run_backward(const std::vector<TensorPtr>& outputs,
                     const std::vector<TensorPtr>& output_grads = {});
    
    /**
     * @brief Register a node in the computation graph
     * 
     * @param output The output tensor
     * @param inputs The input tensors
     * @param backward_fn Function to compute gradients
     * @return The created node
     */
    NodePtr register_node(const TensorPtr& output,
                         const std::vector<TensorPtr>& inputs,
                         BackwardFunctionPtr backward_fn);
    
    /**
     * @brief Get or create node for a tensor
     */
    NodePtr get_node(const TensorPtr& tensor);
    
    /**
     * @brief Clear the computation graph (called after backward)
     */
    void clear_graph();
    
    /**
     * @brief Enable/disable autograd globally
     */
    void set_enabled(bool enabled) { enabled_ = enabled; }
    bool is_enabled() const { return enabled_; }

private:
    Engine() : enabled_(true) {}
    
    // Map from tensor raw pointer to its node
    std::unordered_map<const Tensor*, NodePtr> tensor_to_node_;
    
    // Map from tensor raw pointer to its shared_ptr (for setting gradients)
    std::unordered_map<const Tensor*, TensorPtr> tensor_registry_;
    
    // Temporary gradient accumulation during backward
    std::unordered_map<const Tensor*, TensorPtr> pending_grads_;
    
    bool enabled_;
    
    // Internal helpers
    std::vector<NodePtr> topological_sort(const std::vector<NodePtr>& roots);
    void accumulate_grad(const TensorPtr& tensor, const TensorPtr& grad);
};

} // namespace autograd
} // namespace ops
