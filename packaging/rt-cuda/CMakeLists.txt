cmake_minimum_required(VERSION 3.24)
project(loom_rt_cuda LANGUAGES C CXX)

# Everything of substance is in the shared helper; this file is the two facts that differ per
# backend. This is the rt-vulkan file with `vulkan`/`GGML_VULKAN` replaced -- which is what P4.8a
# claimed a second backend would cost, now spent.
#
# THE ARCHITECTURE LIST IS SET HERE, and "just take ggml's default" is wrong for two separate reasons
# (BACKLOG.md P4.8h). Measured, wheel sizes, all with GGML_CUDA_FA=OFF below:
#
#   ggml's default under CUDA 12.8              148 MB   PTX 5.0-9.0 + cubins 8.6/8.9/12.0a
#   the list below, under CUDA 13.1              88 MB   PTX 7.5/8.0/9.0 + cubins 8.6/8.9/12.0a/12.1a
#   no list at all, under CUDA 13.1             135 MB   TEN cubin sets and NO PTX
#
# The last row is the trap. ggml only picks a default `if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)`, and
# under CUDA 13 CMake defines it first -- so ggml's carefully chosen list is skipped and every
# architecture the toolkit knows gets a real cubin, with no JIT fallback for anything else. "Default"
# is not a stable thing to rely on across toolkits.
#
# WHAT THIS LIST BUYS, against the hardware loom targets. Real cubins for the GPUs people have --
# 8.6 (RTX 30x, and Jetson Orin's 8.7 by binary compatibility), 8.9 (RTX 40x), 12.0a (RTX 50x), 12.1a
# (DGX Spark) -- and PTX for 7.5/8.0/9.0 so a Turing, an A100 or an H100 JITs on first run rather than
# finding nothing. Each real cubin set costs roughly 25-30 MB; PTX is nearly free.
#
# WHAT IT GIVES UP, deliberately: Maxwell 5.0, Pascal 6.1 and Volta 7.0, which CUDA 13 drops and which
# ggml only lists below that version. Accepted -- those cards cannot run current models usefully.
#
# 12.1a REQUIRES CUDA >= 12.9. On 12.8 the `121a-real` entry is silently unavailable and DGX Spark is
# uncovered; there is no 12.x spelling that reaches it, because ggml rewrites every `12X` to `12Xa`
# (Blackwell's FP4 tensor-core instructions are not forward-compatible and cannot be branched on in
# host code), and an `a` cubin runs only on its exact architecture.
# THE LIST DEPENDS ON THE CPU ARCHITECTURE, because the GPUs that appear next to each one barely
# overlap. pip already selects the wheel by platform tag, so this costs the user nothing and keeps the
# aarch64 wheel -- the one going to the most constrained devices -- from carrying desktop kernels.
#
#   x86_64   desktop and workstation. RTX 30x (8.6), RTX 40x (8.9), RTX 50x (12.0a) get real cubins;
#            Turing (7.5), A100 (8.0) and Hopper (9.0) JIT from PTX. No 12.1a: GB10 is an ARM part.
#   aarch64  Jetson and Grace. Orin (8.7) and DGX Spark (12.1a) get real cubins; GH200 (9.0) and
#            Thor JIT from PTX. No 8.9 or 12.0a -- there is no Ada or RTX 50x board on an ARM host.
#
# 8.7 gets its OWN cubin here rather than relying on 8.6 binary compatibility, because Orin is a
# first-class target on this side and the wheel has room once the desktop kernels are gone.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    set(LOOM_RT_CUDA_ARCHS_DEFAULT "80-virtual;87-real;90-virtual;121a-real")
else()
    set(LOOM_RT_CUDA_ARCHS_DEFAULT "75-virtual;80-virtual;86-real;89-real;90-virtual;120a-real")
endif()
set(LOOM_RT_CUDA_ARCHS "${LOOM_RT_CUDA_ARCHS_DEFAULT}"
    CACHE STRING "CUDA architectures for the packaged backend")
set(CMAKE_CUDA_ARCHITECTURES ${LOOM_RT_CUDA_ARCHS} CACHE STRING "" FORCE)
message(STATUS "loom-rt-cuda: ${CMAKE_SYSTEM_PROCESSOR} -> CUDA archs ${LOOM_RT_CUDA_ARCHS}")

# WHERE libcudart AND libcublas COME FROM AT RUN TIME. This package depends on the `nvidia-*-cu12`
# wheels (see pyproject.toml), which unpack to `nvidia/cuda_runtime/lib/` and `nvidia/cublas/lib/` --
# siblings of this package's own directory in site-packages. A relative RPATH reaches them.
#
# Deliberately NOT the other approach. torch preloads its CUDA libraries with ctypes at import time,
# which does not fit here: `loom/__init__.py` finds accelerator packages by SCANNING sys.path and never
# imports them, precisely so a broken accelerator cannot take the base package down. An RPATH needs
# nobody to execute anything.
#
# BUILD_WITH_INSTALL_RPATH so the value survives into the installed artifact; without it CMake relinks
# at install time and the build-tree RPATH is what ends up in the wheel -- which is how the first
# build shipped a path into this machine's conda prefix.
set(CMAKE_INSTALL_RPATH "$ORIGIN:$ORIGIN/../nvidia/cuda_runtime/lib:$ORIGIN/../nvidia/cublas/lib")
set(CMAKE_BUILD_WITH_INSTALL_RPATH ON)

# FlashAttention kernels are UNREACHABLE from loom and are a third of the binary. The engine builds
# attention compositely -- `mul_mat` -> `soft_max_ext` -> `mul_mat`, with `mul_mat_set_prec` on the QK
# product -- and `ggml_flash_attn_ext` appears nowhere in it, so nothing can ever dispatch to them.
# Measured: 148 MB -> 101 MB under CUDA 12.8 with no functional change whatsoever.
#
# Revisit this the day the engine grows an attention primitive that emits FLASH_ATTN_EXT. That is
# scoped, not built, and it would need a device/CPU parity tolerance re-bisect because FA changes the
# reduction order and internal precision.
set(GGML_CUDA_FA OFF CACHE BOOL "" FORCE)

include(${CMAKE_CURRENT_LIST_DIR}/../common/BackendPackage.cmake)

loom_rt_backend_package(
    NAME        cuda
    PACKAGE     loom_rt_cuda
    GGML_OPTION GGML_CUDA
)
