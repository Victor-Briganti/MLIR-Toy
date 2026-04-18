//===--- Dialect.h - Lexer for the Toy Language ---------------------------===//
//
// Part of the MLIR Toy project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Modified by: Victor Briganti in 2026
//
//===----------------------------------------------------------------------===//
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
