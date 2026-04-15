//===--- Dialect.h - Lexer for the Toy Language ---------------------------===//
//
// This file implements the IR Dialect for the Toy language.
//
//===----------------------------------------------------------------------===//

#ifndef TOY_DIALECT_H
#define TOY_DIALECT_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

/// Include the auto-generate header file containing the declaration of the toy
/// dialect.
#include "toy/Dialect.h.inc"

/// Include the auto-generated header file containing the declarations of the
/// toy language.
#define GET_OP_CLASSES
#include "toy/Ops.h.inc"

#endif // TOY_DIALECT_H
