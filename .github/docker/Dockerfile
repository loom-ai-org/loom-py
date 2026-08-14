# Build images for the accelerator packages: manylinux_2_28 with one backend's SDK already in it.
#
# WHY THESE EXIST. Both backend packages need an SDK the manylinux image does not carry, and
# installing it per job is the slow part of the job. CUDA is four dnf packages; Vulkan is worse --
# with no `glslc` present, `cmake/VulkanToolchain.cmake` fetches shaderc and builds it from source,
# which its own message describes as "several minutes, once per build directory". That cost is
# identical on every run, so it belongs in an image built when the pins change rather than in a job
# that runs on every release.
#
# TWO TARGETS RATHER THAN ONE IMAGE. A Vulkan job has no use for CUDA's several gigabytes, and vice
# versa. They share the base layer, so the registry stores the common part once.
#
#   docker build --target cuda   -t ghcr.io/loom-ai-org/manylinux-loom:cuda   .github/docker
#   docker build --target vulkan -t ghcr.io/loom-ai-org/manylinux-loom:vulkan .github/docker
#
# The tag `manylinux-x86_64-image` in each package's [tool.cibuildwheel] is what points at these.

ARG MANYLINUX=quay.io/pypa/manylinux_2_28_x86_64:latest

# ---------------------------------------------------------------------------------------------------
FROM ${MANYLINUX} AS base
# git and python3 are here for the FetchContent dependencies both backends pull (ggml, and for Vulkan
# shaderc's own `utils/git-sync-deps`, which is a Python script).
RUN dnf -y install dnf-plugins-core git python3 && dnf clean all

# ---------------------------------------------------------------------------------------------------
FROM base AS cuda
# 12.9 AND NOT NEWER OR OLDER, and both bounds are real (BACKLOG.md P4.8i):
#   * >= 12.9 because `121a-real` -- DGX Spark -- is unavailable below it, and no 12.x spelling
#     substitutes, since ggml rewrites every `12X` to `12Xa` for Blackwell's FP4 instructions;
#   * < 13 because the `nvidia-*-cu13` runtime wheels this package DEPENDS on do not exist -- they are
#     1.4 KB placeholder sdists at version 0.0.1 -- so a CUDA 13 build has nothing pip can install
#     beside it.
#
# The four components rather than `cuda-toolkit`: this is what nvcc and ggml actually need, and the
# full toolkit is several gigabytes of profilers and samples that no build step opens.
# `cuda-driver-devel` supplies the libcuda.so STUB -- ggml links CUDA::cuda_driver for VMM, and a build
# container has no driver. Without it the configure fails with `Target "ggml-cuda" links to
# CUDA::cuda_driver but the target was not found`, and the tempting fix (GGML_CUDA_NO_VMM=ON) would
# change shipped memory-pool behaviour to work around a build-environment gap.
ARG CUDA_VERSION=12-9
RUN dnf config-manager --add-repo \
        https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo && \
    dnf -y install \
        cuda-nvcc-${CUDA_VERSION} \
        cuda-cudart-devel-${CUDA_VERSION} \
        libcublas-devel-${CUDA_VERSION} \
        cuda-driver-devel-${CUDA_VERSION} && \
    dnf clean all
ENV PATH=/usr/local/cuda/bin:${PATH} \
    CUDACXX=/usr/local/cuda/bin/nvcc

# ---------------------------------------------------------------------------------------------------
FROM base AS vulkan
# The Vulkan LOADER, which ggml links against. Not bundled into the wheel -- see rt-vulkan's
# repair-wheel-command: a loader's job is to find the TARGET machine's installed drivers.
RUN dnf -y install vulkan-loader-devel && dnf clean all

# glslc, prebuilt. THE PINS MUST MATCH cmake/VulkanToolchain.cmake IN THE ENGINE, because that file
# probes what it finds and silently rebuilds shaderc from source when the answer is inadequate -- so a
# stale pin here does not fail, it just makes every job pay the cost this image exists to avoid. It
# probes the FAILURE rather than a version number (a shaderc too old to emit "extension not supported"
# makes ggml's coopmat probe conclude the extension IS supported, and the build dies in conv2d_mm.comp),
# so a mismatch is invisible except in the build time.
ARG SHADERC_TAG=v2025.4
ARG VULKAN_HEADERS_TAG=v1.4.321
RUN dnf -y install cmake ninja-build && dnf clean all && \
    git clone --depth 1 --branch ${VULKAN_HEADERS_TAG} \
        https://github.com/KhronosGroup/Vulkan-Headers.git /tmp/vulkan-headers && \
    cmake -S /tmp/vulkan-headers -B /tmp/vh-build -G Ninja -DCMAKE_INSTALL_PREFIX=/usr/local && \
    cmake --install /tmp/vh-build && \
    git clone --depth 1 --branch ${SHADERC_TAG} https://github.com/google/shaderc.git /tmp/shaderc && \
    python3 /tmp/shaderc/utils/git-sync-deps && \
    cmake -S /tmp/shaderc -B /tmp/shaderc-build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        -DSHADERC_SKIP_TESTS=ON \
        -DSHADERC_SKIP_EXAMPLES=ON \
        -DSHADERC_SKIP_COPYRIGHT_CHECK=ON && \
    cmake --build /tmp/shaderc-build --target install -j"$(nproc)" && \
    rm -rf /tmp/shaderc /tmp/shaderc-build /tmp/vulkan-headers /tmp/vh-build && \
    glslc --version
