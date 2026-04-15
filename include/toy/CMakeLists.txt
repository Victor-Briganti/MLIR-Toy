# Defines the tablegen files that we are looking
set(LLVM_TARGET_DEFINITIONS Ops.td)

# Generates Op declarations
mlir_tablegen(Ops.h.inc -gen-op-decls)
# Generates Op definitions
mlir_tablegen(Ops.cpp.inc -gen-op-defs)

# Generates Op declarations
mlir_tablegen(Dialect.h.inc -gen-dialect-decls -dialect=toy)
# Generates Op definitions
mlir_tablegen(Dialect.cpp.inc -gen-dialect-defs -dialect=toy)

add_public_tablegen_target(ToyOpIncGen)
