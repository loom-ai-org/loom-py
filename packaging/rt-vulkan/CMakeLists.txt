cmake_minimum_required(VERSION 3.24)
project(loom_rt_vulkan LANGUAGES C CXX)

# Everything of substance is in the shared helper; this file is the two facts that differ per
# backend. A CUDA package is this file with `vulkan`/`GGML_VULKAN` replaced by `cuda`/`GGML_CUDA`.
include(${CMAKE_CURRENT_LIST_DIR}/../common/BackendPackage.cmake)

loom_rt_backend_package(
    NAME        vulkan
    PACKAGE     loom_rt_vulkan
    GGML_OPTION GGML_VULKAN
)
