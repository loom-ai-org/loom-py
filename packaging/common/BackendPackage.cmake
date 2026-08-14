# One ggml backend, built as a standalone shared library and packaged as a Python wheel.
#
# WHAT THIS IS FOR (BACKLOG.md P4.8). The base wheel `loom-py-rt` is built with GGML_BACKEND_DL, so
# every backend is a .so discovered at run time. An accelerator is therefore not a second build of
# everything -- it is one library, in one small wheel, that `loom/__init__.py` finds on sys.path.
# `libggml-vulkan.so` is 46.5 MB (44 MB of it compiled SPIR-V) against a ~3 MB CPU-only deployment,
# and CUDA's fat binaries are larger still; a wheel per accelerator would not fit PyPI's 100 MB
# per-file ceiling, and would republish the base on every platform for every backend added.
#
# WHY IT BUILDS GGML AND NOT THE ENGINE. The engine is not involved: nothing in `libggml-vulkan.so`
# references loom, and the base wheel already ships libloom_engine.so and libggml-base.so. This build
# wants exactly one artifact out of the ggml tree.
#
# THE PIN IS THE WHOLE CORRECTNESS ARGUMENT. The .so this produces will be dlopened beside a
# libggml-base.so built by a different invocation, possibly months apart, and ggml promises nothing
# about ABI across revisions. So the revision comes from the engine checkout's own cmake/GgmlPin.cmake
# rather than from a tag written here -- there is no second copy to drift -- and the wheel this
# produces declares `loom-py-rt == <exact version>` so pip cannot pair mismatched libraries either.
#
# These wheels are built FROM THE REPO, not from an sdist: the path below reaches out to the engine
# submodule, which is above the package root. That is deliberate -- a backend wheel is a CI artifact
# built next to the base wheel it must match, and an sdist that could be built standalone would be an
# sdist that could be built against the wrong ggml.

include(FetchContent)

# loom-py/packaging/common/ -> loom-py/vendor/loom.cpp
set(LOOM_ENGINE_DIR ${CMAKE_CURRENT_LIST_DIR}/../../vendor/loom.cpp)
if(NOT EXISTS ${LOOM_ENGINE_DIR}/cmake/GgmlPin.cmake)
    message(FATAL_ERROR
        "The engine submodule is not checked out at ${LOOM_ENGINE_DIR}. A backend package builds "
        "against the engine's pinned ggml revision and cannot substitute another one; run "
        "`git submodule update --init --recursive` in loom-py.")
endif()

# Builds one ggml backend and installs it into the Python package directory.
#
#   NAME        the ggml backend's own name, as it appears in ggml_backend_load_all's list and in the
#               library filename: "vulkan" -> libggml-vulkan.so.
#   PACKAGE     the Python package directory the .so is installed into (loom_rt_vulkan).
#   GGML_OPTION the ggml CMake option that turns it on (GGML_VULKAN, GGML_CUDA, GGML_OPENVINO, ...).
function(loom_rt_backend_package)
    cmake_parse_arguments(ARG "" "NAME;PACKAGE;GGML_OPTION" "" ${ARGN})

    include(${LOOM_ENGINE_DIR}/cmake/GgmlPin.cmake)
    FetchContent_Declare(ggml
        GIT_REPOSITORY ${LOOM_GGML_REPOSITORY}
        GIT_TAG        ${LOOM_GGML_TAG}
    )

    # The same three settings the base wheel uses, and they have to agree: GGML_BACKEND_DL is what
    # makes a backend a standalone loadable library at all (without it the backend is compiled into
    # libggml.so and there is nothing separate to ship), and GGML_NATIVE off is what keeps the result
    # from being tuned to the machine that built the wheel.
    set(BUILD_SHARED_LIBS ON CACHE BOOL "" FORCE)
    set(GGML_BACKEND_DL ON CACHE BOOL "" FORCE)
    set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
    set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(${ARG_GGML_OPTION} ON CACHE BOOL "" FORCE)

    # Vulkan needs a glslc and Vulkan-Headers newer than a stable distribution ships, in two ways that
    # name neither cause. The engine already solved that; this reuses the solution rather than
    # reproducing it, and it is a no-op for every other backend.
    if(ARG_GGML_OPTION STREQUAL "GGML_VULKAN")
        find_package(Python3 COMPONENTS Interpreter REQUIRED)
        include(${LOOM_ENGINE_DIR}/cmake/VulkanToolchain.cmake)
        FetchContent_GetProperties(ggml)
        if(NOT ggml_POPULATED)
            FetchContent_Populate(ggml)
        endif()
        loom_setup_vulkan_toolchain()
        add_subdirectory(${ggml_SOURCE_DIR} ${ggml_BINARY_DIR} EXCLUDE_FROM_ALL)
    else()
        FetchContent_MakeAvailable(ggml)
    endif()

    set(backend_target ggml-${ARG_NAME})
    if(NOT TARGET ${backend_target})
        message(FATAL_ERROR
            "ggml built no target `${backend_target}`. Either ${ARG_GGML_OPTION} did not take effect, "
            "or this machine is missing the backend's SDK -- ggml skips a backend whose toolchain it "
            "cannot find, and does it without failing the configure step.")
    endif()

    # Only the backend .so travels. libggml-base.so is the base wheel's to ship, and shipping a second
    # copy here would put two of them on the same sys.path with no rule about which loads.
    install(TARGETS ${backend_target} LIBRARY DESTINATION ${ARG_PACKAGE})
endfunction()
