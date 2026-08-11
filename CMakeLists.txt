cmake_minimum_required(VERSION 3.24)  # $<LINK_LIBRARY:WHOLE_ARCHIVE,...> below needs 3.24
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
# Static, not loom.cpp's own default of shared (loom.cpp/cmake/Dependencies.cmake builds
# libloom_engine.so so its 129 test binaries share one .so instead of relinking it 129 times -- a
# concern this build doesn't have). A wheel ships `_loom.cpython-*.so` alone; nothing here installs or
# RPATHs libloom_engine.so/libggml*.so beside it, so a shared build produces an import that fails the
# moment the wheel leaves this build tree ("libloom_engine.so: cannot open shared object file"). Static
# linking folds the engine and ggml directly into `_loom`'s .so, leaving only libc/libstdc++/libgcc/
# libm as external deps -- exactly what every manylinux wheel already assumes.
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
add_subdirectory(vendor/loom.cpp EXCLUDE_FROM_ALL)

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
# WHOLE_ARCHIVE, not a plain link: `loom_engine`'s ops (src/ops/primitives_*.cpp) self-register into
# PrimitiveRegistry via a static-initializer global in each translation unit (LOOM_REGISTER_OP,
# include/loom/ops/primitive_registry.h) -- nothing in binding.cpp calls into those files by symbol,
# everything goes through the registry. A plain static-archive link only pulls .o members that satisfy
# an unresolved symbol reference, so the ordinary link silently DROPPED every op-registration file
# wholesale (confirmed by hand: `PrimitiveRegistry: unknown op 'GET_ROWS'` at runtime, from a wheel
# that built and imported cleanly -- the missing ops have no build-time signal at all). WHOLE_ARCHIVE
# forces every .o in libloom_engine.a into the link regardless of whether anything references it, which
# is exactly what a registry of self-registering translation units needs. ggml doesn't need the same
# treatment -- it has no self-registration, everything in it is reached through ordinary symbol calls.
target_link_libraries(_loom PRIVATE $<LINK_LIBRARY:WHOLE_ARCHIVE,loom_engine>)

# Installed beside the Python package, which is where `from . import _loom` looks. The same location
# a development build writes to, so `cmake --build build && python -c "import loom"` works from the
# source tree without an install step.
install(TARGETS _loom LIBRARY DESTINATION loom)
set_target_properties(_loom PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/loom
)
