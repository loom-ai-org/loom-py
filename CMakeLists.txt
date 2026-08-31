cmake_minimum_required(VERSION 3.24)  # $<LINK_LIBRARY:WHOLE_ARCHIVE,...> below needs 3.24

# THE DEPLOYMENT FLOOR, BEFORE project() BECAUSE THAT IS WHEN CMAKE READS IT. Left unset, clang
# targets the macOS of the machine doing the building, so a wheel built on 15.6 carries
# `LC_BUILD_VERSION minos 15.6` and refuses to load on 12 or 14 -- while still being TAGGED
# `macosx_11_0_arm64` by the packaging side, which reads this same number. A wheel whose filename
# promises more than its Mach-O headers deliver is the AVX2 defect's shape once more: it installs,
# and then does not run, on precisely the machines nobody building it owns.
#
# cibuildwheel exports `MACOSX_DEPLOYMENT_TARGET` (11.0 on arm64, 10.13 on x86_64) and CMake reads
# that into this variable, so CI never reaches the defaults below -- they exist so that a wheel
# built by hand on a Mac is the same artifact as the one CI publishes. `uname` rather than `APPLE`
# because `APPLE` is not defined until project(), which is already too late to set this.
execute_process(COMMAND uname -s OUTPUT_VARIABLE LOOM_HOST_OS
                OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
if(LOOM_HOST_OS STREQUAL "Darwin"
   AND NOT DEFINED CMAKE_OSX_DEPLOYMENT_TARGET
   AND NOT DEFINED ENV{MACOSX_DEPLOYMENT_TARGET})
    execute_process(COMMAND uname -m OUTPUT_VARIABLE LOOM_HOST_ARCH
                    OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
    if(CMAKE_OSX_ARCHITECTURES)
        set(LOOM_HOST_ARCH "${CMAKE_OSX_ARCHITECTURES}")
    endif()
    if(LOOM_HOST_ARCH STREQUAL "arm64")
        set(CMAKE_OSX_DEPLOYMENT_TARGET "11.0" CACHE STRING "macOS deployment floor")
    else()
        set(CMAKE_OSX_DEPLOYMENT_TARGET "10.13" CACHE STRING "macOS deployment floor")
    endif()
    message(STATUS "macOS deployment floor defaulted to ${CMAKE_OSX_DEPLOYMENT_TARGET}")
endif()

project(py_loom LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# The engine, as a submodule rather than a copy. Its own tests and tools are off: this build wants
# libloom_engine and nothing else, and building 129 C++ test executables to ship a Python wheel would
# be a strange way to spend a CI minute.
set(LOOM_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LOOM_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
# SHARED, and this reverses a decision that was right for its own reasons -- worth stating both, since
# the failure the old comment recorded is real and is now merely handled rather than avoided.
#
# It used to be static: a wheel shipped `_loom.cpython-*.so` alone, with the engine and ggml folded
# into it, leaving only libc/libstdc++/libgcc/libm as external deps. A shared build back then produced
# "libloom_engine.so: cannot open shared object file" the moment the wheel left the build tree, because
# nothing installed or RPATHed those libraries beside `_loom`. Both halves of that are fixed below --
# CMAKE_INSTALL_RPATH puts `$ORIGIN` on everything, and the install rules ship the .so files into the
# package directory -- which is what auditwheel does to every wheel with a bundled library anyway.
#
# What forces the change is GGML_BACKEND_DL, which refuses to configure without BUILD_SHARED_LIBS
# (BACKLOG.md P4.8). It makes every backend a .so discovered at RUN time, which is what lets one
# arch-tagged base wheel serve every accelerator: `loom-py-rt-vulkan` and `loom-py-rt-cuda` drop a
# `libggml-vulkan.so` / `libggml-cuda.so` into this package's directory and `device="auto"` finds it.
# The alternative -- a full wheel per accelerator, torch's `cu121` shape -- does not fit PyPI's 100 MB
# per-file ceiling: libggml-vulkan.so alone is 46.5 MB and CUDA is larger.
#
# The CPU becomes a plugin too, and GGML_CPU_ALL_VARIANTS splits it per microarchitecture
# (libggml-cpu-haswell.so, -zen4, -sapphirerapids, ...) with the best picked at load time. That is a
# second, unrelated win: the wheel stops being compiled for one -march. GGML_NATIVE must be off for it,
# for the same reason -- a wheel must not be tuned to whatever machine built it.
set(BUILD_SHARED_LIBS ON CACHE BOOL "" FORCE)
set(GGML_BACKEND_DL ON CACHE BOOL "" FORCE)
set(GGML_CPU_ALL_VARIANTS ON CACHE BOOL "" FORCE)
set(GGML_NATIVE OFF CACHE BOOL "" FORCE)

# METAL OFF, AND ONLY ON THE BASE WHEEL -- it ships in `loom-py-rt-metal` instead (P4.11). This line
# is required rather than tidy: ggml defaults `GGML_METAL` to ON whenever APPLE, so without it a
# macOS base wheel carries `libggml-metal.so` AND the backend package carries a second copy, which
# puts two of them on one sys.path with no rule about which loads -- the arrangement
# packaging/README.md forbids for exactly this reason.
#
# THE SIZE ARGUMENT DOES NOT DECIDE THIS ONE, and it is worth saying so because it decides every
# other backend. `libggml-metal.so` is 0.88 MB, not Vulkan's 46.5 MB -- it embeds the `.metal`
# SOURCE via .incbin and the Metal framework compiles it at run time -- so on size alone it could
# simply ship here.
#
# What decides it is DEVICE SELECTION. Metal registers as a GPU-kind device, so it outranks the CPU
# in the hierarchy (ADR-009 / P4.8e) and `device=""` -- "decide for me" -- resolves to it the moment
# it is present. Measured on an M1 Pro, that is a 2.69x WIN on whisper-small and a 5.2x LOSS on
# VITS, whose graph takes 27 CPU round trips because ggml-metal declines a PAD with a nonzero
# leading pad. Shipping it in the base wheel would apply that trade to everyone who typed
# `pip install loom-py-rt`; shipping it as an extra applies it to someone who asked for a GPU.
#
# That is a statement about today's hierarchy on a unified-memory part, not about Metal, and it is
# tracked as its own item -- if the default stops preferring a GPU whose memory is the CPU's, this
# line is worth revisiting, because 0.88 MB is otherwise a very cheap accelerator to have by default.
set(GGML_METAL OFF CACHE BOOL "" FORCE)

# Every shared library this build produces goes to one directory, so the install rule below is a
# directory copy rather than a list of target names -- GGML_CPU_ALL_VARIANTS creates its per-variant
# targets in a loop, and there is no fixed set of names to enumerate.
set(LOOM_RUNTIME_DIR ${CMAKE_BINARY_DIR}/runtime)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${LOOM_RUNTIME_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${LOOM_RUNTIME_DIR})

# `$ORIGIN` on everything: `_loom.so` finds libloom_engine.so beside it, and that finds libggml-base.so
# beside it, wherever site-packages happens to be. BUILD_WITH_INSTALL_RPATH so the libraries in the
# build tree carry the same RPATH -- otherwise a wheel built by scikit-build-core (which installs from
# the build tree) would need a relink step to acquire it.
#
# `$ORIGIN` IS AN ELF TOKEN AND MACH-O DOES NOT EXPAND IT. macOS spells the same idea
# `@loader_path` -- "the directory of the binary doing the loading" -- and a dyld that meets a
# literal `$ORIGIN` does not error, it simply fails to find the library and reports it as missing.
# That is the same import-time `cannot open shared object file` the paragraph above records as
# fixed, arriving through a third door. `MACOSX_RPATH` is on by default (CMP0042), so each .dylib
# here gets an `@rpath/lib....dylib` install name for this to resolve against.
if(APPLE)
    set(LOOM_LOADER_RELATIVE "@loader_path")
else()
    set(LOOM_LOADER_RELATIVE "$ORIGIN")
endif()
set(CMAKE_INSTALL_RPATH "${LOOM_LOADER_RELATIVE}")
set(CMAKE_BUILD_RPATH "${LOOM_LOADER_RELATIVE}")
set(CMAKE_BUILD_WITH_INSTALL_RPATH ON)

add_subdirectory(vendor/loom.cpp EXCLUDE_FROM_ALL)

# No soname version suffix on the ggml libraries, which is not a cosmetic choice: they are built with
# VERSION/SOVERSION, so each lands on disk as libggml-base.so -> .so.0 -> .so.0.16.0, and a wheel is a
# zip, which cannot carry the symlinks. The two links get materialised as full copies -- three
# byte-identical libggml-base.so files, 2.6 MB of a 7.8 MB wheel. Clearing the properties makes each
# library a single unversioned file, and DT_NEEDED records that name because everything here is linked
# in the same build.
#
# What a soname is FOR does not apply here: it lets several versions of a library coexist for
# independently built consumers. Nothing about this package is independently built -- the libraries
# ship in one directory, resolved through `$ORIGIN`, and a backend package pins the base version with
# `==` precisely because ggml's ABI is not stable enough to allow the mixing a soname would permit.
# set_property with NO value, which UNSETS the property. `set_target_properties(... VERSION "")` looks
# like it should do the same and does not: an empty version is still a version, so the library comes
# out named `libggml-base.so.` -- trailing dot -- and the wheel gains a fourth copy instead of losing
# two. Verified by building all three spellings.
foreach(ggml_lib ggml-base ggml)
    if(TARGET ${ggml_lib})
        set_property(TARGET ${ggml_lib} PROPERTY VERSION)
        set_property(TARGET ${ggml_lib} PROPERTY SOVERSION)
    endif()
endforeach()

# pybind11 by FetchContent, deliberately, so `cmake -B build` works in a bare checkout the same way
# the engine's own ggml dependency does. scikit-build-core installs pybind11 as a build requirement
# too; whichever is found first is used.
find_package(pybind11 2.12 QUIET)
if(NOT pybind11_FOUND)
    include(FetchContent)
    FetchContent_Declare(pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v2.13.6
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(pybind11)
endif()

pybind11_add_module(_loom src/binding.cpp)
# A plain link, and the WHOLE_ARCHIVE that used to be here is gone WITH its problem rather than in
# spite of it -- worth recording, because the problem had no build-time signal and would be expensive
# to rediscover. `loom_engine`'s ops (src/ops/primitives_*.cpp) self-register into PrimitiveRegistry
# via a static-initializer global in each translation unit (LOOM_REGISTER_OP), and nothing in
# binding.cpp references those files by symbol -- everything goes through the registry. Linking a
# STATIC libloom_engine.a therefore pulled only the .o members that satisfied an unresolved symbol and
# silently dropped every op-registration file (`PrimitiveRegistry: unknown op 'GET_ROWS'` at run time,
# from a wheel that built and imported cleanly). WHOLE_ARCHIVE forced them all in.
#
# A SHARED libloom_engine.so has no such member selection: every translation unit is already in the
# library, and the dynamic loader runs every initializer in it when `_loom` pulls it in through
# DT_NEEDED. So the ops register for the same reason they do in loom_cli. (WHOLE_ARCHIVE is a
# static-archive concept and is not valid on a shared target, so this is not a choice either way.)
target_link_libraries(_loom PRIVATE loom_engine)

# Installed beside the Python package, which is where `from . import _loom` looks. The same location
# a development build writes to, so `cmake --build build && python -c "import loom"` works from the
# source tree without an install step.
install(TARGETS _loom LIBRARY DESTINATION loom)
set_target_properties(_loom PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/loom
)

# libloom_engine.so and the ggml family, into the package directory beside `_loom` -- which is both
# what `$ORIGIN` resolves to and where ggml is told to look for backends (loom/__init__.py). A
# directory install rather than install(TARGETS): GGML_CPU_ALL_VARIANTS generates its per-variant
# targets in a loop, so the set of names is not knowable here.
#
# BOTH SUFFIXES, and the pair is not symmetric. On macOS the linked libraries -- libloom_engine,
# libggml-base, libggml -- are SHARED targets and come out `.dylib`, so a `*.so*`-only rule shipped
# `_loom.so` alone and the wheel failed at import. The BACKENDS are not `.dylib` there: ggml builds
# each one as a CMake MODULE (`add_library(${backend} MODULE ...)` in its src/CMakeLists.txt), and
# Darwin's `CMAKE_SHARED_MODULE_SUFFIX` is `.so`, which is why ggml's loader looking for `.so`
# unconditionally is correct rather than a bug -- see Epic-08 §4, blocker 4.
#
# Listing both patterns unconditionally rather than under `if(APPLE)`: a Linux build produces no
# `.dylib` for the second pattern to match, so it costs nothing and there is one rule to read.
install(DIRECTORY ${LOOM_RUNTIME_DIR}/
    DESTINATION loom
    FILES_MATCHING
        PATTERN "*.so*"
        PATTERN "*.dylib"
)

# The same libraries into the source tree, so a development build (`cmake --build build`) leaves a
# working `import loom` without an install step -- which is what the source tree's own `_loom.so`
# already provides, and which stops being true the moment the engine is a separate .so. `.gitignore`
# covers `*.so`, so nothing here becomes a tracked file.
add_custom_command(TARGET _loom POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${LOOM_RUNTIME_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/loom
    COMMENT "Copying libloom_engine.so and libggml-*.so beside the Python package"
)
