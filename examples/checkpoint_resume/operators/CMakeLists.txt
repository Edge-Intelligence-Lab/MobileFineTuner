# Keep the verified operator flags scoped to this directory (not the trainer TUs).
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O2")
option(USE_NEW_AUTOGRAD_ENGINE "Use topological sort autograd engine" ON)
option(AUTOGRAD_DEBUG "Enable autograd debug output" OFF)
option(USE_ARENA_ALLOCATOR "Use arena memory allocator" OFF)
option(DISABLE_MEMORY_POOL "Use system allocation for tensors" OFF)
option(USE_BLAS "Enable Accelerate or BLAS kernels" OFF)
option(ENABLE_PROFILING "Enable performance profiling" OFF)
if(DEFINED ENV{OPS_USE_BLAS})
    string(TOUPPER "$ENV{OPS_USE_BLAS}" _blas_env)
    if(_blas_env MATCHES "^(ON|1|TRUE)$")
        set(USE_BLAS ON CACHE BOOL "Enable BLAS via env" FORCE)
    elseif(_blas_env MATCHES "^(OFF|0|FALSE)$")
        set(USE_BLAS OFF CACHE BOOL "Enable BLAS via env" FORCE)
    endif()
endif()
foreach(flag USE_NEW_AUTOGRAD_ENGINE AUTOGRAD_DEBUG USE_ARENA_ALLOCATOR DISABLE_MEMORY_POOL)
    if(${flag})
        add_compile_definitions(${flag})
    endif()
endforeach()
if(ENABLE_PROFILING)
    add_compile_definitions(ENABLE_PROFILING)
    string(APPEND CMAKE_CXX_FLAGS " -pg")
endif()

add_library(operators STATIC
    finetune_ops/core/tensor.cpp
    finetune_ops/core/training_session.cpp
    finetune_ops/core/phase_probe.cpp
    finetune_ops/core/ops.cpp
    finetune_ops/core/autograd_engine.cpp
    finetune_ops/core/backward_functions.cpp
    finetune_ops/core/checkpoint.cpp
    finetune_ops/core/logger.cpp
    finetune_ops/core/memory_manager.cpp
    finetune_ops/core/step_arena.cpp
    finetune_ops/core/utils.cpp
    finetune_ops/core/lm_loss.cpp
    finetune_ops/core/memory_efficient_attention.cpp
    finetune_ops/core/tokenizer_hf.cpp
    finetune_ops/graph/gemma_model.cpp
    finetune_ops/graph/gemma_lora_injector.cpp
    finetune_ops/graph/safetensors_loader.cpp
    finetune_ops/graph/qwen_model.cpp
    finetune_ops/graph/qwen_lora_saver.cpp
    finetune_ops/data/wikitext2_dataset.cpp
    finetune_ops/data/mmlu_dataset.cpp
    finetune_ops/nn/lora_linear.cpp
    finetune_ops/optim/optimizer.cpp
    finetune_ops/optim/adam.cpp
    opt_ops/sharding/parameter_sharder.cpp
)
target_include_directories(operators PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
    PRIVATE finetune_ops opt_ops)
find_package(Threads REQUIRED)
target_link_libraries(operators PUBLIC Threads::Threads)
set_target_properties(operators PROPERTIES ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

if(USE_BLAS)
    if(APPLE)
        find_library(_blas_libs Accelerate)
    else()
        find_package(BLAS)
        set(_blas_libs ${BLAS_LIBRARIES})
    endif()
endif()
if(USE_BLAS AND _blas_libs)
    add_compile_definitions(USE_BLAS)
    target_link_libraries(operators PUBLIC ${_blas_libs})
else()
    if(USE_BLAS)
        message(WARNING "BLAS not found; using pure C++ kernels")
    endif()
    add_compile_definitions(DISABLE_BLAS_COMPLETELY)
endif()

if(BUILD_TESTS)
    foreach(name resume_states training_session phase_probe resume_process)
        add_executable(test_${name} finetune_ops/optim/test_${name}.cpp)
        target_link_libraries(test_${name} PRIVATE operators)
    endforeach()
    add_test(NAME ResumeStates COMMAND test_resume_states)
    add_test(NAME TrainingSession COMMAND test_training_session)
    add_test(NAME PhaseProbe COMMAND test_phase_probe)
    # test_resume_process takes arguments supplied by tests/test_resume_process.py.
endif()
