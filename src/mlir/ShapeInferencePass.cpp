//===--- ShapeInferencePass.cpp - Shape Inference -------------------------===//
//
// Part of the MLIR Toy project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Modified by: Victor Briganti in 2026
//
//===----------------------------------------------------------------------===//
//
// This file implements a Function level pass performing interprocedural
// propagation or array shape through function specialization.
//
//===----------------------------------------------------------------------===//

#include <memory>

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/TypeID.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"

#include "toy/Dialect.h"
#include "toy/Passes.h"
#include "toy/ShapeInferenceInterface.h"

#define DEBUG_TYPE "shape-inference"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace toy;

/// Include the auto-generated definitions for the shape inference interfaces.
#include "toy/ShapeInferenceInterface.cpp.inc"

namespace {
/// The ShapeInferencePass is a pass that performs intra-procedural shape
/// inference.
///
///     Algorithm:
///
///  1) Build a worklist containing all the operations that return a dynamically
///  shaped tensor: these are the operations that need shape inference. 2)
///  Iterate on the worklist:
///    a) find an operation to process: the next ready operation in the worklist
///    has all of its arguments non-generic,
///    b) remove the operation from the worklist,
///    c) remove the operation from the worklist,
///    d) infer the shape of its output from the argument types.
///  3) If the worklist is empty, the algorithm succeeded.
struct ShapeInferencePass
    : public mlir::PassWrapper<ShapeInferencePass, OperationPass<toy::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ShapeInferencePass)
  StringRef getArgument() const override { return "toy-shape-inference"; }

  void runOnOperation() override {
    auto func = getOperation();

    // Populate the worklist with the operations that need shape inference:
    // these are operations that return a dynamic shape.
    llvm::SmallPtrSet<mlir::Operation *, 16> opWorkList;
    func->walk([&](mlir::Operation *op) {
      if (returnsDynamicShape(op)) {
        opWorkList.insert(op);
      }
    });

    // Populate the worklist in the worklist untill all operations have been
    // inferred or no change happened (fix point).
    while (!opWorkList.empty()) {
      // Find the next operation ready for inference, that is an operation with
      // all operands already resolved (non-generic).
      auto nextOp = llvm::find_if(opWorkList, allOperandsInferred);
      if (nextOp == opWorkList.end()) {
        break;
      }

      mlir::Operation *op = *nextOp;
      opWorkList.erase(op);

      // Ask the operation to infer its output shapes
      LDBG("Inferring shape for: " << *op);
      if (auto shapeOp = llvm::dyn_cast<ShapeInference>(op)) {
        shapeOp.inferShape();
      } else {
        op->emitError("Unable to infer shape of operation without shape "
                      "inference interface");
        return signalPassFailure();
      }
    }

    // If the operation worklist isn't emtpy, this indicates a failure.
    if (!opWorkList.empty()) {
      func->emitError("Shape inference failed, ")
          << opWorkList.size() << " operations couldn't be inferred\n";
      signalPassFailure();
    }
  }

  /// A utility method that returns if the given operation has all of its
  /// operands inferred.
  static bool allOperandsInferred(mlir::Operation *op) {
    return llvm::all_of(op->getOperandTypes(), [](mlir::Type resultTy) {
      return llvm::isa<RankedTensorType>(resultTy);
    });
  }

  /// A utility method that returns if the given operation has a dynamically
  /// shape result.
  static bool returnsDynamicShape(mlir::Operation *op) {
    return llvm::any_of(op->getResultTypes(), [](mlir::Type resultTy) {
      return !llvm::isa<RankedTensorType>(resultTy);
    });
  }
};
} // namespace

/// Create a Shape Inference pass.
std::unique_ptr<mlir::Pass> mlir::toy::createShapeInferencePass() {
  return std::make_unique<ShapeInferencePass>();
}
