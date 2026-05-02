//===--- ErrorCode.h - Code Errors for the Toy Compiler -------------------===//
//
// Part of the MLIR Toy project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Modified by: Victor Briganti in 2026
//
//===----------------------------------------------------------------------===//
//
// This file defines the error codes related to each one of the problems related
// to the compiler phases.
//
//===----------------------------------------------------------------------===//

#ifndef TOY_ERROR_CODE_H
#define TOY_ERROR_CODE_H

#define TOY_SUCCESS 0
#define TOY_UNDEFINED -1

#define TOY_INPUT_INVALID 1
#define TOY_PARSE_FAIL 2
#define TOY_GEN_FAIL 3
#define TOY_CLI_PASS_FAIL 4
#define TOY_PIPELINE_PASS_FAIL 5
#define TOY_LLVM_IR_FAIL 6
#define TOY_JIT_CREATION_FAIL 7
#define TOY_TARGET_MACH_FAIL 8

#endif // TOY_ERROR_CODE_H
