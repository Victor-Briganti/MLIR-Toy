//===--- Passes.h - Toy Passes Definition ---------------------------------===//
//
// Part of the MLIR Toy project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Modified by: Victor Briganti in 2026
//
//===----------------------------------------------------------------------===//
//
// This file exposes the entry points to create compiler passes for Toy.
//
//===----------------------------------------------------------------------===//

#ifndef TOY_PASSES_H
#define TOY_PASSES_H

#include <memory>

namespace mlir {
class Pass;

namespace toy {
std::unique_ptr<Pass> createShapeInferencePass();

/// Create a pass for lowering to operations in the `Affine` dialect, for a
/// subset of the Toy IR (e.g. matmul).
std::unique_ptr<Pass> createLowerToAffinePass();

/// Create a pass for lowering `Affine` and `Std` dialects to the LLVM.
std::unique_ptr<Pass> createLowerToLLVMPass();
} // namespace toy
} // namespace mlir

#endif // TOY_PASSES_H
