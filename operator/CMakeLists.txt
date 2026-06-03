cmake_minimum_required(VERSION 3.10)
project(Operators VERSION 2.0.0 LANGUAGES CXX)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# ============================================================================
# Industrial-grade configuration options
# ============================================================================
option(USE_NEW_AUTOGRAD_ENGINE "Enable topological autograd engine" ON)
option(AUTOGRAD_DEBUG "Enable autograd debug logging" OFF)
option(USE_ARENA_ALLOCATOR "Use arena allocator (control RSS growth)" OFF)
option(USE_BLAS "Enable BLAS acceleration (ON=Accelerate/OpenBLAS, OFF=pure C++)" OFF)
option(DISABLE_MEMORY_POOL "Disable custom memory pool; tensors use system malloc/free" OFF)
# Allow override via env (e.g., OPS_USE_BLAS=ON/OFF)
if(DEFINED ENV{OPS_USE_BLAS})
    string(TOUPPER "$ENV{OPS_USE_BLAS}" _OPS_USE_BLAS)
    if(_OPS_USE_BLAS STREQUAL "ON" OR _OPS_USE_BLAS STREQUAL "1" OR _OPS_USE_BLAS STREQUAL "TRUE")
        set(USE_BLAS ON CACHE BOOL "Enable BLAS via env" FORCE)
    elseif(_OPS_USE_BLAS STREQUAL "OFF" OR _OPS_USE_BLAS STREQUAL "0" OR _OPS_USE_BLAS STREQUAL "FALSE")
        set(USE_BLAS OFF CACHE BOOL "Disable BLAS via env" FORCE)
    endif()
endif()
option(BUILD_TESTS "Build test targets" ON)
option(BUILD_EXAMPLES "Build example targets" OFF)
option(ENABLE_PROFILING "Enable profiling" OFF)
option(ENABLE_GPT2_DEBUG_PRINT "Enable GPT-2 forward debug print" OFF)
option(BUILD_LEGACY_CLI "Build legacy top-level CLIs (eval_mmlu/eval_ppl)" OFF)
option(OPERATORS_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)

# ============================================================================
# C++ standard
# ============================================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ============================================================================
# Compiler options
# ============================================================================
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
if(OPERATORS_WARNINGS_AS_ERRORS)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror")
endif()

if(AUTOGRAD_DEBUG)
    add_definitions(-DAUTOGRAD_DEBUG)
    message(STATUS "Autograd debug output enabled")
endif()

if(USE_NEW_AUTOGRAD_ENGINE)
    add_definitions(-DUSE_NEW_AUTOGRAD_ENGINE)
    message(STATUS "New autograd engine enabled (topological order)")
endif()

if(USE_ARENA_ALLOCATOR)
    add_definitions(-DUSE_ARENA_ALLOCATOR)
    message(STATUS "🔧 Arena allocator enabled (controls RSS growth)")
endif()

if(USE_BLAS)
    message(STATUS "🚀 BLAS enabled")
else()
    add_definitions(-DDISABLE_BLAS_COMPLETELY)
    message(STATUS "✅ BLAS disabled (pure C++ memory-first kernels)")
endif()

if(DISABLE_MEMORY_POOL)
    add_definitions(-DDISABLE_MEMORY_POOL)
    message(STATUS "⚠️ Custom memory pool disabled (tensors use system malloc/free for RSS diagnosis)")
endif()

# Control GPT-2 debug printing
if(ENABLE_GPT2_DEBUG_PRINT)
    add_definitions(-DENABLE_GPT2_DEBUG_PRINT)
    message(STATUS "GPT-2 forward debug print: ON")
else()
    message(STATUS "GPT-2 forward debug print: OFF")
endif()

if(ENABLE_PROFILING)
    add_definitions(-DENABLE_PROFILING)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg")
    message(STATUS "Profiling enabled")
endif()

# ============================================================================
# Include directories used while building this target
# ============================================================================
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/finetune_ops
)

# =========================================================================
# Source list
# =========================================================================

set(FINETUNE_SOURCES
    # Core system files
    finetune_ops/core/tensor.cpp
    finetune_ops/core/ops.cpp
    finetune_ops/core/autograd_engine.cpp
    finetune_ops/core/backward_functions.cpp
    finetune_ops/core/logger.cpp
    finetune_ops/core/memory_manager.cpp
    finetune_ops/core/step_arena.cpp
    finetune_ops/core/utils.cpp
    finetune_ops/core/lm_loss.cpp
    finetune_ops/core/memory_efficient_attention.cpp
    finetune_ops/core/performance_monitor.cpp
    
    # GPT-2 finetuning core files
    finetune_ops/core/tokenizer_bpe.cpp
    finetune_ops/graph/gemma_model.cpp
    finetune_ops/graph/gemma_lora_injector.cpp
    finetune_ops/core/tokenizer_gemma.cpp
    finetune_ops/graph/safetensors_loader.cpp
    finetune_ops/graph/gpt2_model.cpp
    finetune_ops/graph/lora_injector.cpp
    # Qwen model and MMLU dataset
    finetune_ops/graph/qwen_model.cpp
    finetune_ops/data/mmlu_dataset.cpp
    
    # Stage B: data loading and training
    finetune_ops/data/wikitext2_dataset.cpp
    finetune_ops/nn/lora_linear.cpp
    finetune_ops/optim/optimizer.cpp
    finetune_ops/optim/adam.cpp
    finetune_ops/optim/trainer.cpp
    
    # Stage C: LoRA save/load
    finetune_ops/graph/lora_saver.cpp
    finetune_ops/optim/gemma_trainer.cpp
)

# =========================================================================
# Create static library
# =========================================================================
add_library(operators STATIC
    ${FINETUNE_SOURCES}
)
add_library(MobileFineTuner::operators ALIAS operators)

# Propagate only the stable finetune surface to dependents.
target_include_directories(operators
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/finetune_ops>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

# =============================================================================
# Modular targets
# =============================================================================
# Using new directory layout (no modules/ references)

# ============================================================================
# BLAS / Accelerate linkage
# ============================================================================
if(USE_BLAS)
    if(APPLE)
        # macOS: use Accelerate Framework
        find_library(ACCELERATE_LIBRARY Accelerate)
        if(ACCELERATE_LIBRARY)
            target_link_libraries(operators PUBLIC ${ACCELERATE_LIBRARY})
            add_definitions(-DUSE_BLAS)
            message(STATUS "✅ Linked Accelerate Framework: ${ACCELERATE_LIBRARY}")
        else()
            message(WARNING "Accelerate not found, falling back to pure C++ implementation")
            add_definitions(-DDISABLE_BLAS_COMPLETELY)
        endif()
    else()
        # Linux/Windows: try OpenBLAS or MKL
        find_package(BLAS)
        if(BLAS_FOUND)
            target_link_libraries(operators PUBLIC ${BLAS_LIBRARIES})
            add_definitions(-DUSE_BLAS)
            message(STATUS "✅ Linked BLAS: ${BLAS_LIBRARIES}")
        else()
            message(WARNING "BLAS not found, falling back to pure C++ implementation")
            add_definitions(-DDISABLE_BLAS_COMPLETELY)
        endif()
    endif()
else()
    message(STATUS "🔧 BLAS disabled, using pure C++ implementation")
endif()

# ============================================================================
# Threading support
# ============================================================================
find_package(Threads REQUIRED)
target_link_libraries(operators PUBLIC Threads::Threads)

# ============================================================================
# Output configuration
# ============================================================================
set_target_properties(operators PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    VERSION ${PROJECT_VERSION}
)

# ============================================================================
# Install rules
# ============================================================================
install(TARGETS operators
    EXPORT OperatorsTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(DIRECTORY
    ${CMAKE_CURRENT_SOURCE_DIR}/include/mobile_finetuner
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING
        PATTERN "*.h"
        PATTERN "*.hpp"
)

install(DIRECTORY
    ${CMAKE_CURRENT_SOURCE_DIR}/finetune_ops/core
    ${CMAKE_CURRENT_SOURCE_DIR}/finetune_ops/data
    ${CMAKE_CURRENT_SOURCE_DIR}/finetune_ops/graph
    ${CMAKE_CURRENT_SOURCE_DIR}/finetune_ops/nn
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/finetune_ops
    FILES_MATCHING
        PATTERN "*.h"
        PATTERN "*.hpp"
)

install(FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/finetune_ops/optim/adam.h
    ${CMAKE_CURRENT_SOURCE_DIR}/finetune_ops/optim/gemma_trainer.h
    ${CMAKE_CURRENT_SOURCE_DIR}/finetune_ops/optim/optimizer.h
    ${CMAKE_CURRENT_SOURCE_DIR}/finetune_ops/optim/trainer.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/finetune_ops/optim
)

install(EXPORT OperatorsTargets
    FILE OperatorsTargets.cmake
    NAMESPACE Operators::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Operators
)

install(EXPORT OperatorsTargets
    FILE MobileFineTunerTargets.cmake
    NAMESPACE MobileFineTuner::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/MobileFineTuner
)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/OperatorsConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/OperatorsConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/OperatorsConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Operators
)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/MobileFineTunerConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/MobileFineTunerConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/MobileFineTunerConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/MobileFineTuner
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/OperatorsConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/OperatorsConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Operators
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/MobileFineTunerConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/MobileFineTunerConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/MobileFineTuner
)

# =============================================================================
# Modular targets
# =============================================================================
# Using new directory layout (no modules/ references)

# ============================================================================
# Test targets
# ============================================================================
if(BUILD_TESTS)
    enable_testing()
    
    # Autograd engine tests
    if(USE_NEW_AUTOGRAD_ENGINE)
        if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/../tests/test_autograd_engine.cpp)
            add_executable(test_autograd_engine
                ../tests/test_autograd_engine.cpp
            )
            target_link_libraries(test_autograd_engine operators)
            add_test(NAME AutogradEngine COMMAND test_autograd_engine)
        endif()
    endif()
    
    # === Stage B test programs ===
    
    # Backward gradient test
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/finetune_ops/optim/test_backward_sanity.cpp)
        add_executable(test_backward_sanity
            finetune_ops/optim/test_backward_sanity.cpp
        )
        target_link_libraries(test_backward_sanity operators)
    endif()
    
    # Simple gradient test
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/finetune_ops/optim/test_simple_grad.cpp)
        add_executable(test_simple_grad
            finetune_ops/optim/test_simple_grad.cpp
        )
        target_link_libraries(test_simple_grad operators)
    endif()
    
    # QKT * softmax backward unit test
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/finetune_ops/optim/test_qkt_softmax_grad.cpp)
        add_executable(test_qkt_softmax_grad
            finetune_ops/optim/test_qkt_softmax_grad.cpp
        )
        target_link_libraries(test_qkt_softmax_grad operators)
        add_test(NAME QKTSoftmaxGrad COMMAND test_qkt_softmax_grad)
    endif()
    
    # RMSNorm forward/backward small-dim unit test (T4)
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/finetune_ops/optim/test_rmsnorm_unit.cpp)
        add_executable(test_rmsnorm_unit
            finetune_ops/optim/test_rmsnorm_unit.cpp
        )
        target_link_libraries(test_rmsnorm_unit operators)
        add_test(NAME RMSNormUnit COMMAND test_rmsnorm_unit)
    endif()
    
    # repeat_kv + softmax backward unit test
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/finetune_ops/optim/test_repeat_kv_softmax_grad.cpp)
        add_executable(test_repeat_kv_softmax_grad
            finetune_ops/optim/test_repeat_kv_softmax_grad.cpp
        )
        target_link_libraries(test_repeat_kv_softmax_grad operators)
        add_test(NAME RepeatKVSoftmaxGrad COMMAND test_repeat_kv_softmax_grad)
    endif()
    
    # T3: single-layer attention-only backward microtest
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/finetune_ops/optim/test_attention_single_layer_backward.cpp)
        add_executable(test_attention_single_layer_backward
            finetune_ops/optim/test_attention_single_layer_backward.cpp
        )
        target_link_libraries(test_attention_single_layer_backward operators)
        add_test(NAME AttentionSingleLayerBackward COMMAND test_attention_single_layer_backward)
    endif()
    
    # LoRALinear gradient test
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/finetune_ops/optim/test_lora_grad.cpp)
        add_executable(test_lora_grad
            finetune_ops/optim/test_lora_grad.cpp
        )
        target_link_libraries(test_lora_grad operators)
    endif()

    # Standalone LoRALinear forward/backward alignment test
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/finetune_ops/nn/test_lora_linear.cpp)
        add_executable(test_lora_linear
            finetune_ops/nn/test_lora_linear.cpp
        )
        target_link_libraries(test_lora_linear operators)
        add_test(NAME LoRALinearUnit COMMAND test_lora_linear)
    endif()
    
    # Cross-Entropy gradient test
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/finetune_ops/optim/test_ce_grad.cpp)
        add_executable(test_ce_grad
            finetune_ops/optim/test_ce_grad.cpp
        )
        target_link_libraries(test_ce_grad operators)
    endif()
    
    # 10-step training convergence check
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/finetune_ops/optim/test_10step_convergence.cpp)
        add_executable(test_10step_convergence
            finetune_ops/optim/test_10step_convergence.cpp
        )
        target_link_libraries(test_10step_convergence operators)
    endif()
    
    # LoRA round-trip test
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/finetune_ops/graph/test_lora_roundtrip.cpp)
        add_executable(test_lora_roundtrip
            finetune_ops/graph/test_lora_roundtrip.cpp
        )
        target_link_libraries(test_lora_roundtrip operators)
        add_test(NAME LoRARoundTrip COMMAND test_lora_roundtrip)
    endif()

    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/finetune_ops/optim/test_gpt2_lora_two_step_smoke.cpp)
        add_executable(test_gpt2_lora_two_step_smoke
            finetune_ops/optim/test_gpt2_lora_two_step_smoke.cpp
        )
        target_link_libraries(test_gpt2_lora_two_step_smoke operators)
        add_test(NAME GPT2LoRATwoStepSmoke COMMAND test_gpt2_lora_two_step_smoke)
    endif()

    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/finetune_ops/graph/test_gemma_lora_roundtrip.cpp)
        add_executable(test_gemma_lora_roundtrip
            finetune_ops/graph/test_gemma_lora_roundtrip.cpp
        )
        target_link_libraries(test_gemma_lora_roundtrip operators)
        add_test(NAME GemmaLoRARoundTrip COMMAND test_gemma_lora_roundtrip)
    endif()

    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/finetune_ops/core/test_memory_efficient_attention_grad.cpp)
        add_executable(test_memory_efficient_attention_grad
            finetune_ops/core/test_memory_efficient_attention_grad.cpp
        )
        target_link_libraries(test_memory_efficient_attention_grad operators)
        add_test(NAME MemoryEfficientAttentionGrad COMMAND test_memory_efficient_attention_grad)
    endif()

    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/finetune_ops/core/test_apply_rope_layout.cpp)
        add_executable(test_apply_rope_layout
            finetune_ops/core/test_apply_rope_layout.cpp
        )
        target_link_libraries(test_apply_rope_layout operators)
        add_test(NAME ApplyRoPELayout COMMAND test_apply_rope_layout)
    endif()

    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/finetune_ops/core/test_autograd_leaf_accum.cpp)
        add_executable(test_autograd_leaf_accum
            finetune_ops/core/test_autograd_leaf_accum.cpp
        )
        target_link_libraries(test_autograd_leaf_accum operators)
        add_test(NAME AutogradLeafAccum COMMAND test_autograd_leaf_accum)
    endif()

    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/finetune_ops/core/test_dtype_mixed_precision.cpp)
        add_executable(test_dtype_mixed_precision
            finetune_ops/core/test_dtype_mixed_precision.cpp
        )
        target_link_libraries(test_dtype_mixed_precision operators)
        add_test(NAME DTypeMixedPrecision COMMAND test_dtype_mixed_precision)
    endif()

    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/finetune_ops/core/test_tensor_slice.cpp)
        add_executable(test_tensor_slice
            finetune_ops/core/test_tensor_slice.cpp
        )
        target_link_libraries(test_tensor_slice operators)
        add_test(NAME TensorSlice COMMAND test_tensor_slice)
    endif()

    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/finetune_ops/core/test_selected_token_lm_loss.cpp)
        add_executable(test_selected_token_lm_loss
            finetune_ops/core/test_selected_token_lm_loss.cpp
        )
        target_link_libraries(test_selected_token_lm_loss operators)
        add_test(NAME SelectedTokenLMLoss COMMAND test_selected_token_lm_loss)
    endif()

    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/finetune_ops/data/test_jsonl_mask_alignment.cpp)
        add_executable(test_jsonl_mask_alignment
            finetune_ops/data/test_jsonl_mask_alignment.cpp
        )
        target_link_libraries(test_jsonl_mask_alignment operators)
        add_test(NAME JsonlMaskAlignment COMMAND test_jsonl_mask_alignment)
    endif()

    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/finetune_ops/graph/test_gemma_config.cpp)
        add_executable(test_gemma_config
            finetune_ops/graph/test_gemma_config.cpp
        )
        target_link_libraries(test_gemma_config operators)
    endif()

    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/finetune_ops/graph/test_gemma_forward.cpp)
        add_executable(test_gemma_forward
            finetune_ops/graph/test_gemma_forward.cpp
        )
        target_link_libraries(test_gemma_forward operators)
    endif()
    
    # Baseline GPT-2 training (full finetune comparison)
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/finetune_ops/optim/test_baseline_10step.cpp)
        add_executable(test_baseline_10step
            finetune_ops/optim/test_baseline_10step.cpp
        )
        target_link_libraries(test_baseline_10step operators)
    endif()
    
    # Fixed-batch comparison test
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/finetune_ops/optim/test_fixed_batch_compare.cpp)
        add_executable(test_fixed_batch_compare
            finetune_ops/optim/test_fixed_batch_compare.cpp
        )
        target_link_libraries(test_fixed_batch_compare operators)
    endif()
    
    message(STATUS "✅ Test targets enabled")
endif()

if(BUILD_LEGACY_CLI)
    # ============================================================================
    # Main program: gpt2_lora_finetune
    # ============================================================================
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/../gpt2_lora_finetune/main.cpp)
        add_executable(gpt2_lora_finetune
            ../gpt2_lora_finetune/main.cpp
        )
        target_link_libraries(gpt2_lora_finetune operators)
    endif()

    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/finetune_ops/optim/train_lora_gemma.cpp)
        add_executable(train_lora_gemma
            finetune_ops/optim/train_lora_gemma.cpp
        )
        target_link_libraries(train_lora_gemma operators)
    endif()

    # MMLU eval
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/../gpt2_lora_finetune/eval_mmlu.cpp)
        add_executable(eval_mmlu
            ../gpt2_lora_finetune/eval_mmlu.cpp
            ../gpt2_lora_finetune/mmlu/mmlu_runner.cpp
        )
        target_include_directories(eval_mmlu PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../gpt2_lora_finetune)
        target_link_libraries(eval_mmlu operators)
    endif()

    # WikiText-2 PPL eval
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/../gpt2_lora_finetune/eval_ppl.cpp)
        add_executable(eval_ppl
            ../gpt2_lora_finetune/eval_ppl.cpp
        )
        target_include_directories(eval_ppl PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../gpt2_lora_finetune)
        target_link_libraries(eval_ppl operators)
    endif()

    # Quick PPL eval
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/../gpt2_lora_finetune/quick_eval_ppl.cpp)
        add_executable(quick_eval_ppl
            ../gpt2_lora_finetune/quick_eval_ppl.cpp
        )
        target_include_directories(quick_eval_ppl PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../gpt2_lora_finetune)
        target_link_libraries(quick_eval_ppl operators)
    endif()

    # Quick PPL eval with LoRA support
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/../gpt2_lora_finetune/quick_eval_lora.cpp)
        add_executable(quick_eval_lora
            ../gpt2_lora_finetune/quick_eval_lora.cpp
        )
        target_include_directories(quick_eval_lora PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../gpt2_lora_finetune)
        target_link_libraries(quick_eval_lora operators)
    endif()
endif()

# ============================================================================
# Configuration summary
# ============================================================================
message(STATUS "")
message(STATUS "============================================")
message(STATUS "Operators configuration summary v${PROJECT_VERSION}")
message(STATUS "============================================")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Stable core:")
message(STATUS "  - Autograd engine: ${USE_NEW_AUTOGRAD_ENGINE}")
message(STATUS "  - BLAS acceleration: ${USE_BLAS}")
message(STATUS "  - Arena allocator: ${USE_ARENA_ALLOCATOR}")
message(STATUS "Developer tools:")
message(STATUS "  - Build tests: ${BUILD_TESTS}")
message(STATUS "  - Profiling: ${ENABLE_PROFILING}")
message(STATUS "============================================")
message(STATUS "")
