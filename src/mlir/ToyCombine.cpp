//===--- ToyCombine.td - Toy High Level Optimizer -------------------------===//
//
// Part of the MLIR Toy project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Modified by: Victor Briganti in 2026
//
//===----------------------------------------------------------------------===//
//
// This file implements a set of simple combiners for optimizing operations in
// the Toy dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "llvm/Support/LogicalResult.h"

#include "toy/Dialect.h"

using namespace mlir;

namespace {
/// Include the patterns defined in the Declarative Rewrite framework.
#include "ToyCombine.cpp.inc"
} // namespace

/// This is an example of a C++ rewrite pattern for the TransposeOs. It
/// optimizes teh following scenarios: tranpose(transpose(x)) -> x
struct SimplifyRedundantTranspose
    : public mlir::OpRewritePattern<toy::TransposeOp> {
  /// We register this pattern to match every toy.transpose in the IR.
  /// The "benefic" is used by the framework to order the patterns and process
  /// them in order of profitability.
  SimplifyRedundantTranspose(mlir::MLIRContext *context)
      : mlir::OpRewritePattern<toy::TransposeOp>(context, 1) {}

  /// This method attempts to match a pattern and rewrite it. The rewriter
  /// argument is the orchestartor of the sequence of rewrites. The pattern is
  /// expected to interact with it to perform any changes to the IR from here.
  llvm::LogicalResult
  matchAndRewrite(toy::TransposeOp op,
                  mlir::PatternRewriter &rewriter) const override {
    // Look through the input of the current transpose.
    mlir::Value transposeInput = op.getOperand();
    toy::TransposeOp transposeInputOp =
        transposeInput.getDefiningOp<toy::TransposeOp>();

    // Input defined by another transpose? If not, no match.
    if (!transposeInputOp) {
      return llvm::failure();
    }

    // Otherwise, we have a redundant tranpose. Use the rewriter.
    rewriter.replaceOp(op, {transposeInputOp->getOperand(0)});
    return llvm::success();
  }
};

/// Register our patterns as "canonicalization" patterns on the TranposeOp so
/// that they can be picked up by Canonicalization framework.
void toy::TransposeOp::getCanonicalizationPatterns(
    mlir::RewritePatternSet &results, mlir::MLIRContext *context) {
  results.add<SimplifyRedundantTranspose>(context);
}

/// Register our patterns as "canonicalization" patterns on the ReshapeOp so
/// that they can be picked up by Canonicalization framework.
void toy::ReshapeOp::getCanonicalizationPatterns(
    mlir::RewritePatternSet &results, mlir::MLIRContext *context) {
  results.add<ReshapeReshapeOpPattern, FoldConstantReshapeOptPattern,
              RedundantReshapeOptPattern>(context);
}
