//===--- ShapeInferenceInterface.h - Shape Inference Interface ------------===//
//
// Part of the MLIR Toy project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Modified by: Victor Briganti in 2026
//
//===----------------------------------------------------------------------===//
//
// This file contains the declaration of the shape inference interfaces defined
// in ShapeInferenceInterface.td
//
//===----------------------------------------------------------------------===//

#ifndef TOY_SHAPE_INFERENCE_INTERFACE_OPS_
#define TOY_SHAPE_INFERENCE_INTERFACE_OPS_

#include "mlir/IR/OpDefinition.h"

namespace mlir {
namespace toy {

// Include the auto-generated declarations.
#include "toy/ShapeInferenceInterface.h.inc"

} // namespace toy
} // namespace mlir

#endif // TOY_SHAPE_INFERENCE_INTERFACE_OPS_