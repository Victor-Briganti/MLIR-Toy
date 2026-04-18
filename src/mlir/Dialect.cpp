//===--- Dialect.cpp - Toy IR Dialect registration in MLIR ----------------===//
//
// This file implements the dialect for the Toy IR: custom type parsing and
// operation verification.
//
//===----------------------------------------------------------------------===//

#include "toy/Dialect.h"

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/LogicalResult.h>
#include <mlir/IR/AffineExpr.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/Types.h>
#include <mlir/Interfaces/FunctionImplementation.h>
#include <mlir/Support/LLVM.h>

using namespace mlir::toy;

#include "toy/Dialect.cpp.inc"

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
}

//===----------------------------------------------------------------------===//
// Toy Operations
//===----------------------------------------------------------------------===//

/// A generalized parser for binary operations. This parses the different forms
/// of 'printBinaryOp' below.
static mlir::ParseResult parseBinaryOp(mlir::OpAsmParser &Parser,
                                       mlir::OperationState &Result) {
  mlir::SmallVector<mlir::OpAsmParser::UnresolvedOperand, 2> Ops;
  mlir::SMLoc OpsLoc = Parser.getCurrentLocation();
  mlir::Type OpType;
  if (Parser.parseOperandList(Ops, 2) ||
      Parser.parseOptionalAttrDict(Result.attributes) ||
      Parser.parseColonType(OpType)) {
    return mlir::failure();
  }

  // If the type is a function type, it contains the input and result types of
  // this operation.
  if (mlir::FunctionType FuncType =
          llvm::dyn_cast<mlir::FunctionType>(OpType)) {
    if (Parser.resolveOperands(Ops, FuncType.getInputs(), OpsLoc,
                               Result.operands)) {
      return mlir::failure();
    }

    Result.addTypes(FuncType.getResults());
    return mlir::success();
  }

  // Otherwise, the parsed type is the type of both operands and results.
  if (Parser.resolveOperands(Ops, OpType, Result.operands)) {
    return mlir::failure();
  }

  Result.addTypes(OpType);
  return mlir::success();
}

/// A generalized printer for bianry operations. It prints in two different
/// forms depending on if all of the types match.
static void printBinaryOp(mlir::OpAsmPrinter &OS, mlir::Operation *Op) {
  OS << " " << Op->getOperands();
  OS.printOptionalAttrDict(Op->getAttrs());
  OS << " : ";

  // If all of the types are of the same, print the type directly.
  mlir::Type ResultType = *Op->result_type_begin();
  if (llvm::all_of(Op->getOperandTypes(),
                   [=](mlir::Type T) { return T == ResultType; })) {
    OS << ResultType;
  }

  // Otherwise, print a function type.
  OS.printFunctionalType(Op->getOperandTypes(), Op->getResultTypes());
}

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

/// Build a constant operation.
/// The builder is passed as an argument, so is the state that this method is
/// expected to fill in order to build the operation.
void ConstantOp::build(mlir::OpBuilder &Builder, mlir::OperationState &State,
                       double Value) {
  auto DataType = RankedTensorType::get({}, Builder.getF64Type());
  auto DataAttribute = DenseElementsAttr::get(DataType, Value);
  ConstantOp::build(Builder, State, DataType, DataAttribute);
}

/// The `OpAsmParser` class provides a collection of methods for parsing various
/// punctuation, as well as attributes, operands, types, etc. Each of these
/// methods returns a `ParseResult`. The class is wrapper around `LogicalResult`
/// that can be converted to a boolean `true` value on failure, or `false` on
/// success. This allows for easily chaining together a set of parser rules.
/// These rules are used to populate an `mlir::OperationState` similarly to the
/// `build` methods described above.
mlir::ParseResult ConstantOp::parse(mlir::OpAsmParser &Parser,
                                    mlir::OperationState &Result) {
  mlir::DenseElementsAttr Value;
  if (Parser.parseOptionalAttrDict(Result.attributes) ||
      Parser.parseAttribute(Value, "value", Result.attributes)) {
    return mlir::failure();
  }

  Result.addTypes(Value.getType());
  return mlir::success();
}

void ConstantOp::print(mlir::OpAsmPrinter &OS) {
  OS << " " << getValue();
  OS.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"value"});
}

/// Verifier for the constant operation. This corresponds to the `let
/// hasVerifier = 1` in the op definition.
llvm::LogicalResult ConstantOp::verify() {
  // If the return type of the constant is not an unranked tensor, the sahpe
  // must match the shape of attribute holding the data.
  auto ResultType =
      llvm::dyn_cast<mlir::RankedTensorType>(getResult().getType());
  if (!ResultType) {
    return mlir::success();
  }

  // Check that the rank of the attribute type matches the rank of the constant
  // result type.
  auto AttrType = llvm::cast<mlir::RankedTensorType>(getValue().getType());
  if (AttrType.getRank() != ResultType.getRank()) {
    return emitOpError("return type must match the one of the attached value "
                       "attribute: ")
           << AttrType.getRank() << " != " << ResultType.getRank();
  }

  // Check that each of the dimenions match between the two types.
  for (int Dim = 0, DimE = static_cast<int>(AttrType.getRank()); Dim < DimE;
       Dim++) {
    size_t DimSz = static_cast<size_t>(Dim);
    if (AttrType.getShape()[DimSz] != ResultType.getShape()[DimSz]) {
      return emitOpError(
                 "return type shape mismatches its attributes at dimension ")
             << DimSz << ": " << AttrType.getShape()[DimSz]
             << " != " << ResultType.getShape()[DimSz];
    }
  }

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// AddOp
//===----------------------------------------------------------------------===//

void AddOp::build(mlir::OpBuilder &Builder, mlir::OperationState &State,
                  mlir::Value LHS, mlir::Value RHS) {
  State.addTypes(UnrankedTensorType::get(Builder.getF64Type()));
  State.addOperands({LHS, RHS});
}

mlir::ParseResult AddOp::parse(mlir::OpAsmParser &Parser,
                               mlir::OperationState &Result) {
  return parseBinaryOp(Parser, Result);
}

void AddOp::print(mlir::OpAsmPrinter &OS) { printBinaryOp(OS, *this); }

//===----------------------------------------------------------------------===//
// MulOp
//===----------------------------------------------------------------------===//

void MulOp::build(mlir::OpBuilder &Builder, mlir::OperationState &State,
                  mlir::Value LHS, mlir::Value RHS) {
  State.addTypes(UnrankedTensorType::get(Builder.getF64Type()));
  State.addOperands({LHS, RHS});
}

mlir::ParseResult MulOp::parse(mlir::OpAsmParser &Parser,
                               mlir::OperationState &Result) {
  return parseBinaryOp(Parser, Result);
}

void MulOp::print(mlir::OpAsmPrinter &OS) { printBinaryOp(OS, *this); }

//===----------------------------------------------------------------------===//
// FuncOp
//===----------------------------------------------------------------------===//

void FuncOp::build(mlir::OpBuilder &Builder, mlir::OperationState &State,
                   llvm::StringRef Name, mlir::FunctionType FuncType,
                   llvm::ArrayRef<mlir::NamedAttribute> Attrs) {
  // FunctionOpInterface provides a convenient `build` method that will populate
  // the state of our FuncOp and create an entry block.
  buildWithEntryBlock(Builder, State, Name, FuncType, Attrs,
                      FuncType.getInputs());
}

mlir::ParseResult FuncOp::parse(mlir::OpAsmParser &Parser,
                                mlir::OperationState &Result) {
  // Dispatch to the FunctionOpInterface provided utility method that parses the
  // function operation.
  auto BuildFuncType =
      [](mlir::Builder &Builder, llvm::ArrayRef<mlir::Type> ArgTypes,
         llvm::ArrayRef<mlir::Type> Results,
         mlir::function_interface_impl::VariadicFlag,
         std::string &) { return Builder.getFunctionType(ArgTypes, Results); };

  return mlir::function_interface_impl::parseFunctionOp(
      Parser, Result, false, getFunctionTypeAttrName(Result.name),
      BuildFuncType, getArgAttrsAttrName(Result.name),
      getResAttrsAttrName(Result.name));
}

void FuncOp::print(mlir::OpAsmPrinter &OS) {
  // Dispatch to the FunctionOpInterface provided utility method that prints the
  // function operations
  mlir::function_interface_impl::printFunctionOp(
      OS, *this, false, getFunctionTypeAttrName(), getArgAttrsAttrName(),
      getResAttrsAttrName());
}

//===----------------------------------------------------------------------===//
// GenericCallOp
//===----------------------------------------------------------------------===//

void GenericCallOp::build(mlir::OpBuilder &Builder, mlir::OperationState &State,
                          StringRef Callee, llvm::ArrayRef<mlir::Value> Args) {
  // Generic call always returns an unranked Tensor initially
  State.addTypes(UnrankedTensorType::get(Builder.getF64Type()));
  State.addOperands(Args);
  State.addAttribute("callee",
                     mlir::SymbolRefAttr::get(Builder.getContext(), Callee));
}

//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

llvm::LogicalResult ReturnOp::verify() {
  // We know that the parent operation is a funciton, because of the 'HasParent'
  // trait attached to the operation definition.
  auto Func = cast<FuncOp>((*this)->getParentOp());

  // ReturnOps can only have a single optional operand
  if (getNumOperands() > 1) {
    return emitOpError() << "expects at most 1 return operand";
  }

  // The operand number and types must match the function signature.
  const auto &Res = Func.getFunctionType().getResults();
  if (getNumOperands() != Res.size()) {
    return emitOpError() << "does not return the same number of values ("
                         << getNumOperands() << ") as the enclosing function ("
                         << Res.size() << ")";
  }

  // If the operation does not have a input, we are done.
  if (!hasOperand()) {
    return mlir::success();
  }

  auto InputType = *operand_type_begin();
  auto ResType = Res.front();

  // Check that the result type of the function matches the operand type.
  if (InputType == ResType || llvm::isa<mlir::UnrankedTensorType>(InputType) ||
      llvm::isa<mlir::UnrankedTensorType>(ResType)) {
    return mlir::success();
  }

  return emitError() << "type of return operand (" << InputType
                     << ") doesn't match function result type (" << ResType
                     << ")";
}

//===----------------------------------------------------------------------===//
// TransposeOp
//===----------------------------------------------------------------------===//

void TransposeOp::build(mlir::OpBuilder &Builder, mlir::OperationState &State,
                        mlir::Value Val) {
  State.addTypes(UnrankedTensorType::get(Builder.getF64Type()));
  State.addOperands(Val);
}

llvm::LogicalResult TransposeOp::verify() {
  auto InType = llvm::dyn_cast<RankedTensorType>(getOperand().getType());
  auto ResType = llvm::dyn_cast<RankedTensorType>(getType());

  if (!InType || !ResType) {
    return mlir::success();
  }

  auto InShape = InType.getShape();
  if (!std::equal(InShape.begin(), InShape.end(),
                  ResType.getShape().rbegin())) {
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
