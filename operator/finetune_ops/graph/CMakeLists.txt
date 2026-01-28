cmake_minimum_required(VERSION 3.10)

# 测试可执行文件（前向对齐 + LoRA 正确性）
add_executable(test_gpt2_forward test_gpt2_forward.cpp)
target_link_libraries(test_gpt2_forward operators)

add_executable(test_lora_correctness test_lora_correctness.cpp)
target_link_libraries(test_lora_correctness operators)

add_executable(test_gemma_config test_gemma_config.cpp)
target_link_libraries(test_gemma_config operators)

add_executable(test_gemma_forward test_gemma_forward.cpp)
target_link_libraries(test_gemma_forward operators)
