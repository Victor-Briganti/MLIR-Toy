# Defines the tablegen files that we are looking
set(LLVM_TARGET_DEFINITIONS Ops.td)
mlir_tablegen(Ops.h.inc -gen-op-decls) # Generates Op declarations
mlir_tablegen(Ops.cpp.inc -gen-op-defs) # Generates Op definitions
mlir_tablegen(Dialect.h.inc -gen-dialect-decls -dialect=toy)
mlir_tablegen(Dialect.cpp.inc -gen-dialect-defs -dialect=toy)
add_public_tablegen_target(ToyOpIncGen)

set(LLVM_TARGET_DEFINITIONS ShapeInferenceInterface.td)
mlir_tablegen(ShapeInferenceInterface.h.inc -gen-op-interface-decls -dialect=toy)
mlir_tablegen(ShapeInferenceInterface.cpp.inc -gen-op-interface-defs -dialect=toy)
add_public_tablegen_target(ToyShapeInferenceInterfaceIncGen)