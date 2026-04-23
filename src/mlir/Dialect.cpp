//===--- Dialect.cpp - Toy IR Dialect registration in MLIR ----------------===//
//
// Part of the MLIR Toy project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Modified by: Victor Briganti in 2026
//
//===----------------------------------------------------------------------===//
//
// This file implements the dialect for the Toy IR: custom type parsing and
// operation verification.
//
//===----------------------------------------------------------------------===//

#include "toy/Dialect.h"

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/LogicalResult.h>
#include <mlir/IR/AffineExpr.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/Region.h>
#include <mlir/IR/TypeRange.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Interfaces/CallInterfaces.h>
#include <mlir/Interfaces/FunctionImplementation.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/InliningUtils.h>

using namespace mlir::toy;

#include "toy/Dialect.cpp.inc"

//===----------------------------------------------------------------------===//
// ToyInlinerInterface
//===----------------------------------------------------------------------===//

/// This class defines the interface for handling inlining with Toy operations.
struct ToyInlinerInterface : public mlir::DialectInlinerInterface {
  using mlir::DialectInlinerInterface::DialectInlinerInterface;

  //++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  // Analysis Hooks
  //++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

  /// All call operations within toy can be inlined.
  bool isLegalToInline(mlir::Operation *, mlir::Operation *, bool) const final {
    return true;
  }

  /// All functions within toy ca be inlined.
  bool isLegalToInline(mlir::Region *, mlir::Region *, bool,
                       mlir::IRMapping &) const final {
    return true;
  }

  /// All operations within toy can be inlined.
  bool isLegalToInline(mlir::Operation *, mlir::Region *, bool,
                       mlir::IRMapping &) const final {
    return true;
  }

  //++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  // Transformation Hooks
  //++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

  /// Handle the given inlined terminator(toy.return) by replacing it with a new
  /// operation as necessary.
  void handleTerminator(mlir::Operation *op,
                        mlir::ValueRange valuesToRepl) const final {
    // Only "toy.return" needs to be handled here.
    auto returnOp = llvm::cast<ReturnOp>(op);

    // Replace the value directly with the return operands.
    assert(returnOp.getNumOperands() == valuesToRepl.size());
    for (const auto &it : llvm::enumerate(returnOp.getOperands())) {
      valuesToRepl[it.index()].replaceAllUsesWith(it.value());
    }
  }

  /// Attempts to materialize a conversion for a type mismatch between a call
  /// from this dialect, and a callable region. This method should generate an
  /// operation that takes 'input' as the only operand, and produces a single
  /// result of 'resultType'. If a conversion can not be generated, nullptr
  /// should be returned.
  mlir::Operation *
  materializeCallConversion(mlir::OpBuilder &builder, mlir::Value input,
                            mlir::Type resultType,
                            mlir::Location conversionLoc) const final {
    return CastOp::create(builder, conversionLoc, resultType, input);
  }
};

//===----------------------------------------------------------------------===//
// ToyDialect
//===----------------------------------------------------------------------===//

/// Dialect initialization, the instance will be owned by the context. This is
/// the point of registration of types and operations for the dialect.
void ToyDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "toy/Ops.cpp.inc"
      >();
  addInterface<ToyInlinerInterface>();
}

//===----------------------------------------------------------------------===//
// Toy Operations
//===----------------------------------------------------------------------===//

/// A generalized parser for binary operations. This parses the different forms
/// of 'printBinaryOp' below.
static mlir::ParseResult parseBinaryOp(mlir::OpAsmParser &parser,
                                       mlir::OperationState &result) {
  mlir::SmallVector<mlir::OpAsmParser::UnresolvedOperand, 2> ops;
  mlir::SMLoc opsLoc = parser.getCurrentLocation();
  mlir::Type opType;
  if (parser.parseOperandList(ops, 2) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(opType)) {
    return mlir::failure();
  }

  // If the type is a function type, it contains the input and result types of
  // this operation.
  if (mlir::FunctionType funcType =
          llvm::dyn_cast<mlir::FunctionType>(opType)) {
    if (parser.resolveOperands(ops, funcType.getInputs(), opsLoc,
                               result.operands)) {
      return mlir::failure();
    }

    result.addTypes(funcType.getResults());
    return mlir::success();
  }

  // Otherwise, the parsed type is the type of both operands and results.
  if (parser.resolveOperands(ops, opType, result.operands)) {
    return mlir::failure();
  }

  result.addTypes(opType);
  return mlir::success();
}

/// A generalized printer for bianry operations. It prints in two different
/// forms depending on if all of the types match.
static void printBinaryOp(mlir::OpAsmPrinter &os, mlir::Operation *op) {
  os << " " << op->getOperands();
  os.printOptionalAttrDict(op->getAttrs());
  os << " : ";

  // If all of the types are of the same, print the type directly.
  mlir::Type resultType = *op->result_type_begin();
  if (llvm::all_of(op->getOperandTypes(),
                   [=](mlir::Type T) { return T == resultType; })) {
    os << resultType;
  }

  // Otherwise, print a function type.
  os.printFunctionalType(op->getOperandTypes(), op->getResultTypes());
}

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

/// Build a constant operation.
/// The builder is passed as an argument, so is the state that this method is
/// expected to fill in order to build the operation.
void ConstantOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                       double Value) {
  auto dataType = RankedTensorType::get({}, builder.getF64Type());
  auto dataAttribute = DenseElementsAttr::get(dataType, Value);
  ConstantOp::build(builder, state, dataType, dataAttribute);
}

/// The `OpAsmParser` class provides a collection of methods for parsing various
/// punctuation, as well as attributes, operands, types, etc. Each of these
/// methods returns a `ParseResult`. The class is wrapper around `LogicalResult`
/// that can be converted to a boolean `true` value on failure, or `false` on
/// success. This allows for easily chaining together a set of parser rules.
/// These rules are used to populate an `mlir::OperationState` similarly to the
/// `build` methods described above.
mlir::ParseResult ConstantOp::parse(mlir::OpAsmParser &parser,
                                    mlir::OperationState &result) {
  mlir::DenseElementsAttr value;
  if (parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseAttribute(value, "value", result.attributes)) {
    return mlir::failure();
  }

  result.addTypes(value.getType());
  return mlir::success();
}

void ConstantOp::print(mlir::OpAsmPrinter &p) {
  p << " " << getValue();
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"value"});
}

/// Verifier for the constant operation. This corresponds to the `let
/// hasVerifier = 1` in the op definition.
llvm::LogicalResult ConstantOp::verify() {
  // If the return type of the constant is not an unranked tensor, the sahpe
  // must match the shape of attribute holding the data.
  auto resultType =
      llvm::dyn_cast<mlir::RankedTensorType>(getResult().getType());
  if (!resultType) {
    return mlir::success();
  }

  // Check that the rank of the attribute type matches the rank of the constant
  // result type.
  auto attrType = llvm::cast<mlir::RankedTensorType>(getValue().getType());
  if (attrType.getRank() != resultType.getRank()) {
    return emitOpError("return type must match the one of the attached value "
                       "attribute: ")
           << attrType.getRank() << " != " << resultType.getRank();
  }

  // Check that each of the dimenions match between the two types.
  for (int dim = 0, dimE = static_cast<int>(attrType.getRank()); dim < dimE;
       dim++) {
    size_t dimSz = static_cast<size_t>(dim);
    if (attrType.getShape()[dimSz] != resultType.getShape()[dimSz]) {
      return emitOpError(
                 "return type shape mismatches its attributes at dimension ")
             << dimSz << ": " << attrType.getShape()[dimSz]
             << " != " << resultType.getShape()[dimSz];
    }
  }

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// BroadcastOp
//===----------------------------------------------------------------------===//

void BroadcastOp::inferShape() {
  auto arrayTy = llvm::cast<RankedTensorType>(getOperand().getType());
  llvm::SmallVector<int64_t, 2> dims(llvm::reverse(arrayTy.getShape()));
  getResult().setType(RankedTensorType::get(dims, arrayTy.getElementType()));
}

llvm::LogicalResult BroadcastOp::verify() {
  auto inType = llvm::dyn_cast<RankedTensorType>(getOperand().getType());
  auto resType = llvm::dyn_cast<RankedTensorType>(getType());

  if (!inType || !resType) {
    return mlir::success();
  }

  auto inShape = inType.getShape();
  auto resShape = resType.getShape();

  // NumPy broadcasting rules:
  // Compare shapes element-wise from right to left.
  // Two dimensions are compatible when:
  // 1. They are equal, or
  // 2. one of them is 1.
  if (resShape.size() < inShape.size()) {
    return emitError() << "result rank must be >= input rank";
  }

  int i = static_cast<int>(inShape.size()) - 1;
  int j = static_cast<int>(resShape.size()) - 1;

  while (i >= 0) {
    if (inShape[static_cast<size_t>(i)] != 1 &&
        inShape[static_cast<size_t>(i)] != resShape[static_cast<size_t>(j)]) {
      return emitError() << "input dimension " << i << " of size "
                         << inShape[static_cast<size_t>(i)]
                         << " is incompatible with result dimension " << j
                         << " of size " << resShape[static_cast<size_t>(j)];
    }
    i--;
    j--;
  }

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// AddOp
//===----------------------------------------------------------------------===//

void AddOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                  mlir::Value lhs, mlir::Value rhs) {
  state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
  state.addOperands({lhs, rhs});
}

mlir::ParseResult AddOp::parse(mlir::OpAsmParser &parser,
                               mlir::OperationState &result) {
  return parseBinaryOp(parser, result);
}

void AddOp::print(mlir::OpAsmPrinter &p) { printBinaryOp(p, *this); }

void AddOp::inferShape() { getResult().setType(getLhs().getType()); }

//===----------------------------------------------------------------------===//
// MulOp
//===----------------------------------------------------------------------===//

void MulOp::build(mlir::OpBuilder &builder, mlir::OperationState &s,
                  mlir::Value lhs, mlir::Value rhs) {
  s.addTypes(UnrankedTensorType::get(builder.getF64Type()));
  s.addOperands({lhs, rhs});
}

mlir::ParseResult MulOp::parse(mlir::OpAsmParser &parser,
                               mlir::OperationState &result) {
  return parseBinaryOp(parser, result);
}

void MulOp::print(mlir::OpAsmPrinter &p) { printBinaryOp(p, *this); }

void MulOp::inferShape() { getResult().setType(getLhs().getType()); }

//===----------------------------------------------------------------------===//
// CastOp
//===----------------------------------------------------------------------===//

void CastOp::inferShape() { getResult().setType(getInput().getType()); }

/// Returns true if the given set of input and result types are compatible with
/// this cast operation. This is required by the `CastOpInferface` to verify
/// this operation and provide other additional utilities.
bool CastOp::areCastCompatible(mlir::TypeRange inputs,
                               mlir::TypeRange outputs) {
  if (inputs.size() != 1 || outputs.size() != 1) {
    return false;
  }

  // The inputs must be Tensors with the same element type.
  auto input = llvm::dyn_cast<TensorType>(inputs.front());
  auto output = llvm::dyn_cast<TensorType>(outputs.front());
  if (!input || !output || input.getElementType() != output.getElementType()) {
    return false;
  }

  // The shape is required to match if both types are ranked
  return !input.hasRank() || !output.hasRank() || input == output;
}

//===----------------------------------------------------------------------===//
// FuncOp
//===----------------------------------------------------------------------===//

void FuncOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                   llvm::StringRef name, mlir::FunctionType funcType,
                   llvm::ArrayRef<mlir::NamedAttribute> attrs) {
  // FunctionOpInterface provides a convenient `build` method that will populate
  // the state of our FuncOp and create an entry block.
  buildWithEntryBlock(builder, state, name, funcType, attrs,
                      funcType.getInputs());
}

mlir::ParseResult FuncOp::parse(mlir::OpAsmParser &parser,
                                mlir::OperationState &result) {
  // Dispatch to the FunctionOpInterface provided utility method that parses the
  // function operation.
  auto buildFuncType =
      [](mlir::Builder &builder, llvm::ArrayRef<mlir::Type> argTypes,
         llvm::ArrayRef<mlir::Type> results,
         mlir::function_interface_impl::VariadicFlag,
         std::string &) { return builder.getFunctionType(argTypes, results); };

  return mlir::function_interface_impl::parseFunctionOp(
      parser, result, false, getFunctionTypeAttrName(result.name),
      buildFuncType, getArgAttrsAttrName(result.name),
      getResAttrsAttrName(result.name));
}

void FuncOp::print(mlir::OpAsmPrinter &p) {
  // Dispatch to the FunctionOpInterface provided utility method that prints the
  // function operations
  mlir::function_interface_impl::printFunctionOp(
      p, *this, false, getFunctionTypeAttrName(), getArgAttrsAttrName(),
      getResAttrsAttrName());
}

//===----------------------------------------------------------------------===//
// GenericCallOp
//===----------------------------------------------------------------------===//

void GenericCallOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                          StringRef callee,
                          llvm::ArrayRef<mlir::Value> arguments) {
  // Generic call always returns an unranked Tensor initially
  state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
  state.addOperands(arguments);
  state.addAttribute("callee",
                     mlir::SymbolRefAttr::get(builder.getContext(), callee));
}

/// Return the callee of the generic call operation, this isrequired by the call
/// interface.
mlir::CallInterfaceCallable GenericCallOp::getCallableForCallee() {
  return (*this)->getAttrOfType<SymbolRefAttr>("callee");
}

/// Set the callee for the generic call operation, this is required by the call
/// interface.
void GenericCallOp::setCalleeFromCallable(mlir::CallInterfaceCallable callee) {
  return (*this)->setAttr("callee", callee.dyn_cast<SymbolRefAttr>());
}

/// Get the argument operands to the called function, this is required by the
/// call interface.
mlir::Operation::operand_range GenericCallOp::getArgOperands() {
  return getInputs();
}

/// Get the argument operands to the called function as a mutable range, rthis
/// is required by teh call interface.
mlir::MutableOperandRange GenericCallOp::getArgOperandsMutable() {
  return getInputsMutable();
}

//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

llvm::LogicalResult ReturnOp::verify() {
  // We know that the parent operation is a funciton, because of the 'HasParent'
  // trait attached to the operation definition.
  auto func = cast<FuncOp>((*this)->getParentOp());

  // ReturnOps can only have a single optional operand
  if (getNumOperands() > 1) {
    return emitOpError() << "expects at most 1 return operand";
  }

  // The operand number and types must match the function signature.
  const auto &res = func.getFunctionType().getResults();
  if (getNumOperands() != res.size()) {
    return emitOpError() << "does not return the same number of values ("
                         << getNumOperands() << ") as the enclosing function ("
                         << res.size() << ")";
  }

  // If the operation does not have a input, we are done.
  if (!hasOperand()) {
    return mlir::success();
  }

  auto inputType = *operand_type_begin();
  auto resType = res.front();

  // Check that the result type of the function matches the operand type.
  if (inputType == resType || llvm::isa<mlir::UnrankedTensorType>(inputType) ||
      llvm::isa<mlir::UnrankedTensorType>(resType)) {
    return mlir::success();
  }

  return emitError() << "type of return operand (" << inputType
                     << ") doesn't match function result type (" << resType
                     << ")";
}

//===----------------------------------------------------------------------===//
// TransposeOp
//===----------------------------------------------------------------------===//

void TransposeOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                        mlir::Value input) {
  state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
  state.addOperands(input);
}

void TransposeOp::inferShape() {
  auto arrayTy = llvm::cast<RankedTensorType>(getOperand().getType());
  llvm::SmallVector<int64_t, 2> dims(llvm::reverse(arrayTy.getShape()));
  getResult().setType(RankedTensorType::get(dims, arrayTy.getElementType()));
}

llvm::LogicalResult TransposeOp::verify() {
  auto inType = llvm::dyn_cast<RankedTensorType>(getOperand().getType());
  auto resType = llvm::dyn_cast<RankedTensorType>(getType());

  if (!inType || !resType) {
    return mlir::success();
  }

  auto inShape = inType.getShape();
  if (!std::equal(inShape.begin(), inShape.end(),
                  resType.getShape().rbegin())) {
    return emitError()
           << "expected result shape to be a transpose of the input";
  }

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// TrableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "toy/Ops.cpp.inc"
