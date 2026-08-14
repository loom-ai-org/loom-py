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

# TWO LAYOUTS, because a container only sees the package directory.
#
# In the repo, this file sits in loom-py/packaging/common/ and the engine is two levels up at
# vendor/loom.cpp -- which is what makes the ggml pin unforgeable, since it is read from the engine
# checkout rather than copied here.
#
# `cibuildwheel` mounts ONLY the package directory as /project, so neither `../common` nor
# `../../vendor` exists inside the build container. `packaging/stage.py` therefore assembles a
# self-contained copy: this file and the engine's own cmake modules land side by side in the staged
# package's `cmake/` directory. Nothing is duplicated in the REPO -- the copy exists for the duration
# of one build and is made from the submodule, so there is still exactly one source of truth.
#
# Staged layout wins when present, because that is the one a build container is looking at.
if(EXISTS ${CMAKE_CURRENT_LIST_DIR}/GgmlPin.cmake)
    set(LOOM_ENGINE_CMAKE_DIR ${CMAKE_CURRENT_LIST_DIR})
else()
    set(LOOM_ENGINE_CMAKE_DIR ${CMAKE_CURRENT_LIST_DIR}/../../vendor/loom.cpp/cmake)
endif()
if(NOT EXISTS ${LOOM_ENGINE_CMAKE_DIR}/GgmlPin.cmake)
    message(FATAL_ERROR
        "No GgmlPin.cmake at ${LOOM_ENGINE_CMAKE_DIR}. Building from the repo needs the engine "
        "submodule (`git submodule update --init --recursive`); building in a container needs the "
        "package staged first (`python packaging/stage.py cuda <dir>`).")
endif()

# Builds one ggml backend and installs it into the Python package directory.
#
#   NAME        the ggml backend's own name, as it appears in ggml_backend_load_all's list and in the
#               library filename: "vulkan" -> libggml-vulkan.so.
#   PACKAGE     the Python package directory the .so is installed into (loom_rt_vulkan).
#   GGML_OPTION the ggml CMake option that turns it on (GGML_VULKAN, GGML_CUDA, GGML_OPENVINO, ...).
function(loom_rt_backend_package)
    cmake_parse_arguments(ARG "" "NAME;PACKAGE;GGML_OPTION" "" ${ARGN})

    include(${LOOM_ENGINE_CMAKE_DIR}/GgmlPin.cmake)
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

    # Populate-then-add_subdirectory rather than FetchContent_MakeAvailable, for EVERY backend, and
    # `EXCLUDE_FROM_ALL` is the load-bearing word. MakeAvailable brings ggml's OWN `install()` rules
    # into this project, so `pip wheel` collected them alongside the one target that was asked for:
    # the first CUDA wheel built here was 297 MB and contained `libggml-cuda.so` TWICE -- once at
    # `loom_rt_cuda/` where the install() below puts it, and once at `bin/` where ggml's own rule does
    # -- plus a `lib/libggml-base.so` that the note above says must never travel, because two of them
    # on one sys.path have no rule about which loads. EXCLUDE_FROM_ALL drops ggml's install rules and
    # leaves only ours.
    #
    # This was previously the Vulkan branch's behaviour only, because Vulkan needed Populate for an
    # unrelated reason (a glslc and Vulkan-Headers newer than a stable distribution ships, which the
    # engine already solved and this reuses). CUDA was the first backend to take the other branch and
    # the first to find the bug -- so the branches now differ only in the toolchain step, which is the
    # only thing that is genuinely per-backend.
    FetchContent_GetProperties(ggml)
    if(NOT ggml_POPULATED)
        FetchContent_Populate(ggml)
    endif()
    if(ARG_GGML_OPTION STREQUAL "GGML_VULKAN")
        find_package(Python3 COMPONENTS Interpreter REQUIRED)
        include(${LOOM_ENGINE_CMAKE_DIR}/VulkanToolchain.cmake)
        loom_setup_vulkan_toolchain()
    endif()
    add_subdirectory(${ggml_SOURCE_DIR} ${ggml_BINARY_DIR} EXCLUDE_FROM_ALL)

    set(backend_target ggml-${ARG_NAME})
    if(NOT TARGET ${backend_target})
        message(FATAL_ERROR
            "ggml built no target `${backend_target}`. Either ${ARG_GGML_OPTION} did not take effect, "
            "or this machine is missing the backend's SDK -- ggml skips a backend whose toolchain it "
            "cannot find, and does it without failing the configure step.")
    endif()

    # EXCLUDE_FROM_ALL above took the whole ggml directory out of the default build, this one target
    # included -- so `ninja` had nothing to do and the install below failed looking for a library that
    # was never compiled. Putting exactly one target back is the point: everything ggml would otherwise
    # build and install stays excluded, and the artifact this package exists for gets built.
    set_target_properties(${backend_target} PROPERTIES EXCLUDE_FROM_ALL FALSE)

    # The SONAME the base wheel actually provides. This has to match the root CMakeLists.txt, which
    # unsets VERSION/SOVERSION on the ggml libraries because a wheel is a zip and a zip cannot carry a
    # symlink -- so the base ships one `libggml-base.so`, not the usual .so -> .so.0 -> .so.0.19.0
    # chain. Built with ggml's defaults instead, this backend records `NEEDED libggml-base.so.0`,
    # nothing provides that name, and the dlopen fails.
    #
    # The failure is silent, which is why this is worth the comment: ggml logs a backend that fails to
    # load at a level the binding drops, so the whole symptom is an accelerator that does not appear in
    # `loom.devices()`. It cost a build cycle to find with `ctypes.CDLL` on the shipped file.
    #
    # `set_property` with NO value, which UNSETS. `set_target_properties(... VERSION "")` looks
    # equivalent and is not -- an empty version is still a version, and the library comes out named
    # `libggml-base.so.` with a trailing dot (recorded in BACKLOG.md P4.8a).
    foreach(ggml_lib ggml-base ggml)
        if(TARGET ${ggml_lib})
            set_property(TARGET ${ggml_lib} PROPERTY VERSION)
            set_property(TARGET ${ggml_lib} PROPERTY SOVERSION)
        endif()
    endforeach()

    # Only the backend .so travels. libggml-base.so is the base wheel's to ship, and shipping a second
    # copy here would put two of them on the same sys.path with no rule about which loads.
    install(TARGETS ${backend_target} LIBRARY DESTINATION ${ARG_PACKAGE})
endfunction()
