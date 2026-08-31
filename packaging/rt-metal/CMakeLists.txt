cmake_minimum_required(VERSION 3.24)
project(loom_rt_metal LANGUAGES C CXX)

# Everything of substance is in the shared helper; this file is the two facts that differ per
# backend. Staged layout first (packaging/stage.py assembles it for a build container, which only
# mounts this directory), then the in-repo layout.
if(EXISTS ${CMAKE_CURRENT_LIST_DIR}/cmake/BackendPackage.cmake)
    include(${CMAKE_CURRENT_LIST_DIR}/cmake/BackendPackage.cmake)
else()
    include(${CMAKE_CURRENT_LIST_DIR}/../common/BackendPackage.cmake)
endif()

loom_rt_backend_package(
    NAME        metal
    PACKAGE     loom_rt_metal
    GGML_OPTION GGML_METAL
)
