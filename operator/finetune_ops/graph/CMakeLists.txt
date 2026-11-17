# Graph sources and tests are registered by operator/CMakeLists.txt.
# Keep one build entrypoint so standalone test lists cannot drift out of sync.
if(NOT TARGET operators)
    message(FATAL_ERROR "Configure MobileFineTuner with cmake -S operator -B <build-dir>")
endif()
