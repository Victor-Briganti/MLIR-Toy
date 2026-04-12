# Find LLVM and MLIR
find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVM_DIR: ${LLVM_DIR}")
message(STATUS "Using MLIR_DIR: ${MLIR_DIR}")

# Align C++ standard with LLVM if not already set
if (NOT DEFINED CMAKE_CXX_STANDARD)
  if (LLVM_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD ${LLVM_CXX_STANDARD})
  else()
    set(CMAKE_CXX_STANDARD 17)
  endif()
endif()

# Add LLVM and MLIR CMake module paths
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")

include(AddLLVM)
include(TableGen)
include(AddMLIR)

# Add include directories and definitions
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})

# Macro to help linking LLVM components
macro(mlirtoy_link_llvm_components target)
  llvm_map_components_to_libnames(llvm_libs ${ARGN})
  target_link_libraries(${target} PRIVATE ${llvm_libs})
endmacro()
