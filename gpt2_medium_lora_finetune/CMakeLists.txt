cmake_minimum_required(VERSION 3.16)
project(gpt2_medium_lora_finetune_cli LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(BUILD_TESTS OFF CACHE BOOL "Disable operator tests in app builds" FORCE)
set(BUILD_EXAMPLES OFF CACHE BOOL "Disable operator examples in app builds" FORCE)
add_subdirectory(../operator ${CMAKE_BINARY_DIR}/operator)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../operator)

add_executable(train
    src/main.cpp
)
target_link_libraries(train PRIVATE operators)

add_executable(eval_mmlu
    src/mmlu/mmlu_runner.cpp
    src/eval_mmlu.cpp
)
target_include_directories(eval_mmlu PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(eval_mmlu PRIVATE operators)

# WikiText-2 PPL evaluation
add_executable(eval_ppl
    src/eval_ppl.cpp
)
target_link_libraries(eval_ppl PRIVATE operators)
