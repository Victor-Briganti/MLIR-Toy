cmake_minimum_required(VERSION 3.10)

project(MlirToy
        VERSION
        0.0.1
        DESCRIPTION "MLIR Toy language project"
        LANGUAGES CXX C)

include(cmake/PreventInSourceBuilds.cmake)
include(cmake/LLVM.cmake)
include(cmake/StandardConfiguration.cmake)
include(cmake/CompilerWarnings.cmake)
include(cmake/StaticAnalyzer.cmake)
include(CTest)

# Create the options and warnings targets
add_library(mlirtoy_options INTERFACE)
add_library(mlirtoy_warnings INTERFACE)

# Configure the project with the cxx_standard that we had set
target_compile_features(mlirtoy_options INTERFACE cxx_std_${CMAKE_CXX_STANDARD})
mlirtoy_enable_clang_tidy(mlirtoy_options ON)

# Add the warnings configuration to the project
mlirtoy_set_project_warnings(mlirtoy_warnings)

# Added the warnings and options as a interface
add_library(mlirtoy::mlirtoy_warnings ALIAS mlirtoy_warnings)
add_library(mlirtoy::mlirtoy_options ALIAS mlirtoy_options)

# Add the source directory
add_subdirectory(include/toy)
add_subdirectory(src)
add_subdirectory(app)
add_subdirectory(test)
