cmake_minimum_required(VERSION 3.24)
project(loom_rt_cuda LANGUAGES C CXX)

# Everything of substance is in the shared helper; this file is the two facts that differ per
# backend. This is the rt-vulkan file with `vulkan`/`GGML_VULKAN` replaced -- which is what P4.8a
# claimed a second backend would cost, now spent.
#
# Note what is NOT here: no CUDA architecture list. ggml picks a default when
# CMAKE_CUDA_ARCHITECTURES is unset, and a WHEEL wants that default rather than this machine's GPU --
# the engine's own builds pass `-DCMAKE_CUDA_ARCHITECTURES=120` because the workstation has exactly
# one Blackwell card and a multi-arch build would be larger for nothing, but a published wheel is the
# opposite case. Set it explicitly in CI when the arch coverage becomes a decision (BACKLOG.md P4.8c:
# one architecture is 59 MB stripped, so the 100 MB ceiling constrains how many can travel together).
include(${CMAKE_CURRENT_LIST_DIR}/../common/BackendPackage.cmake)

loom_rt_backend_package(
    NAME        cuda
    PACKAGE     loom_rt_cuda
    GGML_OPTION GGML_CUDA
)
