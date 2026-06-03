cmake_minimum_required(VERSION 3.16)
project(QwenLoRAUnified CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Shared operator library root
get_filename_component(OPERATORS_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../operator" ABSOLUTE)
set(BUILD_TESTS OFF CACHE BOOL "Disable operator tests in app builds" FORCE)
set(BUILD_EXAMPLES OFF CACHE BOOL "Disable operator examples in app builds" FORCE)
add_subdirectory("${OPERATORS_ROOT}" operators_build)

# Executables: wikitext & mmlu
add_executable(train_wikitext
    src/wikitext_main.cpp
)
target_link_libraries(train_wikitext PRIVATE operators)

add_executable(train_mmlu
    src/mmlu_main.cpp
)
target_link_libraries(train_mmlu PRIVATE operators)

if(APPLE)
    find_library(ACCELERATE_LIBRARY Accelerate)
    if(ACCELERATE_LIBRARY)
        target_link_libraries(train_wikitext PRIVATE ${ACCELERATE_LIBRARY})
        target_link_libraries(train_mmlu PRIVATE ${ACCELERATE_LIBRARY})
    endif()
endif()
