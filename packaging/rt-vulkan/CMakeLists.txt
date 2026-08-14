cmake_minimum_required(VERSION 3.24)
project(loom_rt_vulkan LANGUAGES C CXX)

# Everything of substance is in the shared helper; this file is the two facts that differ per
# backend. A CUDA package is this file with `vulkan`/`GGML_VULKAN` replaced by `cuda`/`GGML_CUDA`.
# Staged layout first (packaging/stage.py assembles it for a build container, which only
# mounts this directory), then the in-repo layout.
if(EXISTS ${CMAKE_CURRENT_LIST_DIR}/cmake/BackendPackage.cmake)
    include(${CMAKE_CURRENT_LIST_DIR}/cmake/BackendPackage.cmake)
else()
    include(${CMAKE_CURRENT_LIST_DIR}/../common/BackendPackage.cmake)
endif()

loom_rt_backend_package(
    NAME        vulkan
    PACKAGE     loom_rt_vulkan
    GGML_OPTION GGML_VULKAN
)
