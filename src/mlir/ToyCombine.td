//===- ToyCombine.td - Pattern Match Optimizations for Toy -*- tablegen -*-===//
//
// Part of the MLIR Toy project, under the Apache License v2.0 with LLVM 
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Modified by: Victor Briganti in 2026
//
//===----------------------------------------------------------------------===//
//
// Defines language-specific pattern match optimizations for Toy using 
// Declarative Rewrite Rules (DRR) specified using TableGen record.
//
//===----------------------------------------------------------------------===//

#ifndef TOY_COMBINE
#define TOY_COMBINE

include "mlir/IR/PatternBase.td"
include "toy/Ops.td"

/// Note: The DRR definition used for defining patterns is shown below:
///
/// class Pattern<
///   dag sourcePattern, list<dag>, resultPatterns,
///   list<dag> additionalConstraints = [],
///   list<dag> supplementalPatterns = [],
///   dag benefitsAdded = (addBenefit 0)
/// >;

//===----------------------------------------------------------------------===//
// Basic Pattern-Match and Rewrite
//===----------------------------------------------------------------------===//

// Reshape(Reshape(x)) = Reshape(x)
def ReshapeReshapeOpPattern : Pat<(ReshapeOp(ReshapeOp $arg)),
                                  (ReshapeOp $arg)>;

#endif // TOY_COMBINE
