# --------------------------------------------------------------------------- #
# AST
# --------------------------------------------------------------------------- #

add_library(AST parser/AST.cpp)
add_library(mlirtoy::AST ALIAS AST)
mlirtoy_link_llvm_components(AST Support Core)
target_link_libraries(
  AST
  PRIVATE
  mlirtoy_options
  mlirtoy_warnings)
target_include_directories(
  AST
  ${WARNING_GUARD}
  PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>)
target_compile_features(
  AST
  PUBLIC
  cxx_std_${CMAKE_CXX_STANDARD})
set_target_properties(
  AST
  PROPERTIES
  VERSION
  ${PROJECT_VERSION}
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN YES)

# --------------------------------------------------------------------------- #
# Dialect
# --------------------------------------------------------------------------- #

add_library(Dialect mlir/Dialect.cpp)
add_library(mlirtoy::Dialect ALIAS Dialect)
add_dependencies(Dialect ToyOpIncGen) # ToyOpIncGen needs to be build first
mlirtoy_link_llvm_components(Dialect)
mlirtoy_link_mlir_libs(
  Dialect
  MLIRIR
  MLIRSideEffectInterfaces
  MLIRCallInterfaces
  MLIRFunctionInterfaces
  MLIRBytecodeReader)
target_link_libraries(
  Dialect
  PRIVATE
  mlirtoy_options
  mlirtoy_warnings)
target_include_directories(
  Dialect
  ${WARNING_GUARD}
  PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>)
target_compile_features(
  Dialect
  PUBLIC
  cxx_std_${CMAKE_CXX_STANDARD})
set_target_properties(
  Dialect
  PROPERTIES
  VERSION
  ${PROJECT_VERSION}
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN YES)

# --------------------------------------------------------------------------- #
# MLIRGen
# --------------------------------------------------------------------------- #

add_library(MLIRGen mlir/MLIRGen.cpp)
add_library(mlirtoy::MLIRGen ALIAS MLIRGen)
mlirtoy_link_llvm_components(MLIRGen)
mlirtoy_link_mlir_libs(
  MLIRGen
  MLIRIR
  MLIRParser
  MLIRSupport)
add_dependencies(MLIRGen ToyOpIncGen) # ToyOpIncGen needs to be build first
target_link_libraries(
  MLIRGen
  PRIVATE
  mlirtoy_options
  mlirtoy_warnings)
target_include_directories(
  MLIRGen
  ${WARNING_GUARD}
  PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>)
target_compile_features(
  MLIRGen
  PUBLIC
  cxx_std_${CMAKE_CXX_STANDARD})
set_target_properties(
  MLIRGen
  PROPERTIES
  VERSION
  ${PROJECT_VERSION}
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN YES)
