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
  MLIRCastInterfaces
  MLIRFunctionInterfaces
  MLIRBytecodeReader)
target_link_libraries(
  Dialect
  PRIVATE
  mlirtoy_options
  mlirtoy_warnings)

# --------------------------------------------------------------------------- #
# MLIRGen
# --------------------------------------------------------------------------- #

add_library(MLIRGen mlir/MLIRGen.cpp)
add_library(mlirtoy::MLIRGen ALIAS MLIRGen)
mlirtoy_link_llvm_components(MLIRGen)
mlirtoy_link_mlir_libs(
  MLIRGen
  MLIRIR
  MLIRSupport)
add_dependencies(MLIRGen ToyOpIncGen) # ToyOpIncGen needs to be build first
target_link_libraries(
  MLIRGen
  PRIVATE
  mlirtoy_options
  mlirtoy_warnings)

# --------------------------------------------------------------------------- #
# Shape Inference Pass
# --------------------------------------------------------------------------- #

add_library(ShapeInferencePass mlir/ShapeInferencePass.cpp)
add_library(mlirtoy::ShapeInferencePass ALIAS ShapeInferencePass)
mlirtoy_link_llvm_components(ShapeInferencePass)
mlirtoy_link_mlir_libs(
  ShapeInferencePass
  MLIRIR
  MLIRSupport)
add_dependencies(ShapeInferencePass ToyOpIncGen)
target_link_libraries(
  ShapeInferencePass
  PRIVATE
  mlirtoy_options
  mlirtoy_warnings)

# --------------------------------------------------------------------------- #
# ToyCombine Tablegen
# --------------------------------------------------------------------------- #

# TableGen needs the project include directory to find toy/Ops.td
set(LLVM_TARGET_DEFINITIONS mlir/ToyCombine.td)
mlir_tablegen(ToyCombine.cpp.inc -gen-rewriters "-I${PROJECT_SOURCE_DIR}/include")
add_public_tablegen_target(ToyCombineIncGen)

# --------------------------------------------------------------------------- #
# ToyCombine
# --------------------------------------------------------------------------- #

add_library(ToyCombine mlir/ToyCombine.cpp)
add_library(mlirtoy::ToyCombine ALIAS ToyCombine)
mlirtoy_link_llvm_components(ToyCombine)
mlirtoy_link_mlir_libs(
  ToyCombine
  MLIRIR
  MLIRSupport)
add_dependencies(
  ToyCombine
  ToyOpIncGen
  ToyCombineIncGen
  ToyShapeInferenceInterfaceIncGen)
target_link_libraries(
  ToyCombine
  PRIVATE
  mlirtoy_options
  mlirtoy_warnings)

# Add the binary directory for the generated .inc files
target_include_directories(
  ToyCombine
  PRIVATE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>)

# --------------------------------------------------------------------------- #
# Lowering to Affine
# --------------------------------------------------------------------------- #

add_library(LowerToAffineLoops mlir/LowerToAffineLoops.cpp)
add_library(mlirtoy::LowerToAffineLoops ALIAS LowerToAffineLoops)
mlirtoy_link_llvm_components(LowerToAffineLoops)
mlirtoy_link_mlir_libs(
  LowerToAffineLoops
  MLIRAffineDialect
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRMemRefDialect
  MLIRSupport)
add_dependencies(
  LowerToAffineLoops
  ToyOpIncGen
  ToyCombineIncGen
  ToyShapeInferenceInterfaceIncGen)
target_link_libraries(
  LowerToAffineLoops
  PRIVATE
  mlirtoy_options
  mlirtoy_warnings)

# Add the binary directory for the generated .inc files
target_include_directories(
  LowerToAffineLoops
  PRIVATE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>)

# --------------------------------------------------------------------------- #
# Lowering to LLVM
# --------------------------------------------------------------------------- #

add_library(LowerToLLVM mlir/LowerToLLVM.cpp)
add_library(mlirtoy::LowerToLLVM ALIAS LowerToLLVM)
mlirtoy_link_llvm_components(LowerToLLVM)
mlirtoy_link_mlir_libs(
  LowerToLLVM
  MLIRAffineDialect
  MLIRAffineToStandard
  MLIRArithDialect
  MLIRArithToLLVM
  MLIRControlFlowDialect
  MLIRControlFlowToLLVM
  MLIRFuncDialect
  MLIRFuncToLLVM
  MLIRIR
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
  MLIRMemRefDialect
  MLIRMemRefToLLVM
  MLIRSCFDialect
  MLIRSCFToControlFlow
  MLIRSupport)
add_dependencies(
  LowerToLLVM
  ToyOpIncGen
  ToyCombineIncGen
  ToyShapeInferenceInterfaceIncGen)
target_link_libraries(
  LowerToLLVM
  PRIVATE
  mlirtoy_options
  mlirtoy_warnings)

# Add the binary directory for the generated .inc files
target_include_directories(
  LowerToLLVM
  PRIVATE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>)
