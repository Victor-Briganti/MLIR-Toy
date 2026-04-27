//===--- LowerToAffineLoops.h - Partial Lowering from Toy to Affine -------===//
//
// Part of the MLIR Toy project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Modified by: Victor Briganti in 2026
//
//===----------------------------------------------------------------------===//
//
// This file implements a partial lowering of Toy operations to a combination of
// affine loops memref operations and standard operations. This lowering expects
// that all calls have been inlined, and all shapes have been resolved.
//
//===----------------------------------------------------------------------===//

#include <memory>

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include <cstdint>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/LogicalResult.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/TypeID.h>
#include <mlir/Transforms/DialectConversion.h>

#include "toy/Dialect.h"
#include "toy/Passes.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// ToyToAffine RewritePatterns
//===----------------------------------------------------------------------===//

/// Convert the given RankedTensorType into the corresponding MemRefType.
static mlir::MemRefType convertTensorToMemRef(mlir::RankedTensorType type) {
  return mlir::MemRefType::get(type.getShape(), type.getElementType());
}

/// Insert an allocation and deallocation for the given MemRefType.
static mlir::Value insertAllocAndDealloc(mlir::MemRefType type,
                                         mlir::Location loc,
                                         mlir::PatternRewriter &rewriter) {
  auto alloc = rewriter.create<mlir::memref::AllocaOp>(loc, type);
  auto dealloc = rewriter.create<mlir::memref::AllocaOp>(loc, type);

  // Make sure to allocate at the beginning of the block.
  auto *parentBlock = alloc->getBlock();
  alloc->moveBefore(&parentBlock->front());

  // Make sure to deallocate at the end of the block.
  // NOTE: This is only fine because the Toy language has no control flow.
  dealloc->moveAfter(&parentBlock->front());

  return alloc;
}

/// This defines the function type used to process an iteration of a lowered
/// loop. It takes as input an OpBuilder, an range of memRefOperands
/// corresponding to the operands of the input operation, of the input
/// operation, and the range of loop induction variables for the iteration. It
/// returns a value to store at the current index of the iteration.
using LoopIterationFn = mlir::function_ref<mlir::Value(
    mlir::OpBuilder &rewriter, mlir::ValueRange memRefOperands,
    mlir::ValueRange loopIvs)>;

static void lowerOpToLoops(mlir::Operation *op, mlir::ValueRange operands,
                           mlir::PatternRewriter &rewriter,
                           LoopIterationFn processIteration) {
  auto tensorType =
      llvm::cast<mlir::RankedTensorType>(*op->result_type_begin());
  Location loc = op->getLoc();

  // Insert an allocation and deallocation for the result of this operation.
  mlir::MemRefType memRefTYpe = convertTensorToMemRef(tensorType);
  mlir::Value alloc = insertAllocAndDealloc(memRefTYpe, loc, rewriter);

  // Create a nest of affine lops, with one loop per dimension of the shape.
  // The buildAffineLoopNest function takes a callback that is used to construct
  // the body of the innermost loop given a builder, a location and a range of
  // loop induction variables.
  llvm::SmallVector<int64_t, 4> lowerBounds(tensorType.getRank(), 0);
  llvm::SmallVector<int64_t, 4> steps(tensorType.getRank(), 1);

  affine::buildAffineLoopNest(
      rewriter, loc, lowerBounds, tensorType.getShape(), steps,
      [&](mlir::OpBuilder &nestedBuilder, mlir::Location loc,
          mlir::ValueRange ivs) {
        // Call the processing function with the rewriter, the memref operands,
        // and the loop induction variables. This function will return the value
        // to store at the current index.
        mlir::Value valueToStore =
            processIteration(nestedBuilder, operands, ivs);
        nestedBuilder.create<affine::AffineStoreOp>(loc, valueToStore, alloc,
                                                    ivs);
      });

  // Replace this operation with the genretad alloc.
  rewriter.replaceOp(op, alloc);
}

namespace {
//===----------------------------------------------------------------------===//
// ToyToAffine RewritePatterns: Binary operations
//===----------------------------------------------------------------------===//

template <typename BinaryOp, typename LoweredBinaryOp>
struct BinaryOpLowering : public mlir::ConversionPattern {
  BinaryOpLowering(MLIRContext *ctx)
      : mlir::ConversionPattern(BinaryOp::getOperationName(), 1, ctx) {}

  llvm::LogicalResult
  matchAndRewrite(mlir::Operation *op, llvm::ArrayRef<mlir::Value> operands,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    Location loc = op->getLoc();
    lowerOpToLoops(
        op, operands, rewriter,
        [loc](mlir::OpBuilder &builder, mlir::ValueRange memRefOperands,
              mlir::ValueRange loopIvs) {
          // Generate na adaptor for the remapped operands of the BinaryOp. This
          // allows for using the nice named accessors that are generated by the
          // ODS
          typename BinaryOp::Adaptor binaryAdaptor(memRefOperands);

          // Generate loads for the element of 'lhs' and 'rhs' at the inner
          // loop.
          auto loadedLhs = builder.create<affine::AffineLoadOp>(
              loc, binaryAdaptor.getLhs(), loopIvs);
          auto loadedRhs = builder.create<affine::AffineLoadOp>(
              loc, binaryAdaptor.getRhs(), loopIvs);

          // Create the binary operation performed on the loaded values.
          return builder.create<LoweredBinaryOp>(loc, loadedLhs, loadedRhs);
        });

    return llvm::success();
  }
};

using AddOpLowering = BinaryOpLowering<toy::AddOp, arith::AddFOp>;
using MulOpLowering = BinaryOpLowering<toy::MulOp, arith::MulFOp>;

//===----------------------------------------------------------------------===//
// ToyToAffine RewritePatterns: Constant operations
//===----------------------------------------------------------------------===//

struct ConstantOpLowering : public OpRewritePattern<toy::ConstantOp> {
  using OpRewritePattern<toy::ConstantOp>::OpRewritePattern;

  llvm::LogicalResult
  matchAndRewrite(toy::ConstantOp op,
                  mlir::PatternRewriter &rewriter) const final {
    mlir::DenseElementsAttr constantValue = op.getValue();
    Location loc = op.getLoc();

    // When lowering the constant operation, we allocate and assing the constant
    // values to a corresponding memref allocation.
    auto tensorType = llvm::cast<mlir::RankedTensorType>(op.getType());
    mlir::MemRefType memRefType = convertTensorToMemRef(tensorType);
    mlir::Value alloc = insertAllocAndDealloc(memRefType, loc, rewriter);

    // We will be generating constant indices up-to the largest dimension.
    // Create these cosntants up-front to avoid large amounts of redundant
    // operations.
    llvm::ArrayRef<int64_t> valueShape = memRefType.getShape();
    llvm::SmallVector<mlir::Value, 8> constantIndices;

    if (!valueShape.empty()) {
      for (auto i : llvm::seq<int64_t>(0, *llvm::max_element(valueShape))) {
        constantIndices.push_back(
            rewriter.create<arith::ConstantIndexOp>(loc, i));
      }
    } else {
      // This is the case of a tensor of rank 0.
      constantIndices.push_back(
          rewriter.create<arith::ConstantIndexOp>(loc, 0));
    }

    // The constant operation represents a multi-dimensional cosntant, so we
    // will need to generate a store for each of the elements. The following
    // functor recursively walks the dimensions of the cosntant shape,
    // generating a store when the recursion hits the base case.
    llvm::SmallVector<mlir::Value, 2> indices;
    auto valueIt = constantValue.value_begin<FloatAttr>();
    std::function<void(uint64_t)> storeElements = [&](uint64_t dimension) {
      // The last dimension is the base case of the recursion, at this point we
      // store the element at the given index.
      if (dimension == valueShape.size()) {
        rewriter.create<affine::AffineStoreOp>(
            loc, rewriter.create<arith::ConstantOp>(loc, *valueIt++), alloc,
            llvm::ArrayRef(indices));
        return;
      }

      // Otherwise, iterate over the current dimension and add the indices to
      // the list
      for (uint64_t i = 0, e = valueShape[dimension]; i != e; ++i) {
        indices.push_back(constantIndices[i]);
        storeElements(dimension + 1);
        indices.pop_back();
      }
    };

    // Start the element storing recursion from the first dimension
    storeElements(0);

    // Replace this operation with the generated alloc.
    rewriter.replaceOp(op, alloc);
    return llvm::success();
  }
};

//===----------------------------------------------------------------------===//
// ToyToAffine RewritePatterns: Func operations
//===----------------------------------------------------------------------===//

struct FuncOpLowering : public OpConversionPattern<toy::FuncOp> {
  using OpConversionPattern<toy::FuncOp>::OpConversionPattern;

  llvm::LogicalResult
  matchAndRewrite(toy::FuncOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    // We only lower the main function as we expect that all other functions
    // have been inlined.
    if (op.getName() != "main") {
      return llvm::failure();
    }

    // Verify that the given main has no inputs and results.
    if (op.getNumArguments() || op.getFunctionType().getNumResults()) {
      return rewriter.notifyMatchFailure(op, [](mlir::Diagnostic &diag) {
        diag << "expected 'main to have 0 inputs and 0 results";
      });
    }

    // Create a new non-toy function, with the same region
    auto func = rewriter.create<mlir::func::FuncOp>(op.getLoc(), op.getName(),
                                                    op.getFunctionType());
    rewriter.inlineRegionBefore(op.getRegion(), func.getBody(), func.end());
    rewriter.eraseOp(op);
    return llvm::success();
  }
};

//===----------------------------------------------------------------------===//
// ToyToAffine RewritePatterns: Print operations
//===----------------------------------------------------------------------===//

struct PrintOpLowering : public OpConversionPattern<toy::PrintOp> {
  using OpConversionPattern<toy::PrintOp>::OpConversionPattern;

  llvm::LogicalResult
  matchAndRewrite(toy::PrintOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    // We don't lower "toy.print" in this pass, but we need to update its
    // operands.
    rewriter.modifyOpInPlace(op,
                             [&] { op->setOperands(adaptor.getOperands()); });
    return llvm::success();
  }
};

//===----------------------------------------------------------------------===//
// ToyToAffine RewritePatterns: Return operations
//===----------------------------------------------------------------------===//

struct ReturnOpLowering : public OpConversionPattern<toy::ReturnOp> {
  using OpConversionPattern<toy::ReturnOp>::OpConversionPattern;

  llvm::LogicalResult
  matchAndRewrite(toy::ReturnOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    // During this lowering, we expect that all function calls have been
    // inlined.
    if (op.hasOperand()) {
      return llvm::failure();
    }

    // We lower "toy.return" directly to "func.return"
    rewriter.replaceOpWithNewOp<func::ReturnOp>(op);
    return llvm::success();
  }
};

//===----------------------------------------------------------------------===//
// ToyToAffine RewritePatterns: Transpose operations
//===----------------------------------------------------------------------===//

struct TransposeOpLowering : public ConversionPattern {
  TransposeOpLowering(mlir::MLIRContext *ctx)
      : ConversionPattern(toy::TransposeOp::getOperationName(), 1, ctx) {}

  llvm::LogicalResult
  matchAndRewrite(Operation *op, llvm::ArrayRef<mlir::Value> operands,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    Location loc = op->getLoc();
    lowerOpToLoops(
        op, operands, rewriter,
        [loc](OpBuilder &builder, mlir::ValueRange memRefOperands,
              mlir::ValueRange loopIvs) {
          // Generate na adaptor for the remapped operands of the TransposeOp.
          // This allows for using the nice named accessors that are generated
          // by the ODS
          toy::TransposeOpAdaptor transposeAdaptor(memRefOperands);
          mlir::Value input = transposeAdaptor.getInput();

          // Transpose the elements by generating a load from the reverse
          // indices.
          llvm::SmallVector<mlir::Value, 2> reverseIvs(llvm::reverse(loopIvs));
          return builder.create<affine::AffineLoadOp>(loc, input, reverseIvs);
        });

    return llvm::success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// ToyToAffineLoweringPass
//===----------------------------------------------------------------------===//

/// This is a partial lowering to affine loops of the toy operations that are
/// computationally intensive (like mamul for example) while keeping the rest of
/// the code in the Toy dialect.
namespace {
struct ToyToAffineLoweringPass
    : public PassWrapper<ToyToAffineLoweringPass,
                         mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ToyToAffineLoweringPass)
  llvm::StringRef getArgument() const override { return "toy-to-affine"; }

  void getDependedndDialects(mlir::DialectRegistry &registry) const {
    registry.insert<affine::AffineDialect, func::FuncDialect,
                    memref::MemRefDialect>();
  }

  void runOnOperation() final {
    // The first thing to define is the conversion target. This will define the
    // final target for the lowering
    ConversionTarget target(getContext());

    // We define the specific operations, or dialects, that are legal targets
    // for this lowering. In our case, we are lowering to a combination of the
    // `Affine`, `Arith`, `Func` and `MemRef` dialects.
    target.addLegalDialect<affine::AffineDialect, BuiltinDialect,
                           arith::ArithDialect, func::FuncDialect,
                           memref::MemRefDialect>();

    // We also define the Toy dialect as Illegal so that the conversion will
    // fail if any of these operations are *not* converted. Given that we
    // actually want a partial lowering, we explicitly mark the Toy operations
    // that don't want to lower, `toy.print`, as `legal`. `toy.print` will still
    // need its operands to  be updated though (as we convert from TensorType to
    // MemRefType), se we only treat it as `legal` if its operands are legal.
    target.addIllegalDialect<toy::ToyDialect>();
    target.addDynamicallyLegalOp<toy::PrintOp>([](toy::PrintOp op) {
      return llvm::none_of(op->getOperandTypes(), [](mlir::Type type) {
        return llvm::isa<mlir::TensorType>(type);
      });
    });

    // Now that the conversion target has been defined, we just need to provide
    // the set of patterns that will lower the TOy operaionts.
    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<AddOpLowering, ConstantOpLowering, FuncOpLowering,
                 MulOpLowering, AddOpLowering, ReturnOpLowering,
                 TransposeOpLowering, PrintOpLowering>(&getContext());

    // With the target and rewrite patterns defined, we can now attempt the
    // conversion. The conversion will signal failure if any of our `illegal`
    // operations were not converted successfully.
    if (llvm::failed(applyPartialConversion(getOperation(), target,
                                            std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace

/// Create a pass for lowering operations in the `Affine` and `Std` dialects,
/// for a subset of the Toy IR (e.g. matmul).
std::unique_ptr<mlir::Pass> mlir::toy::createLowerToAffinePass() {
  return std::make_unique<ToyToAffineLoweringPass>();
}