//===--- MLIRGen.cpp - MLIR Generation from the Toy AST -------------------===//
//
// Part of the MLIR Toy project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Modified by: Victor Briganti in 2026
//
//===----------------------------------------------------------------------===//
//
// This file implements a simple IR generation targeting MLIR from a Module AST
// for the Toy language.
//
//===----------------------------------------------------------------------===//

#include <numeric>
#include <vector>

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/LogicalResult.h"

#include "toy/AST.h"
#include "toy/Dialect.h"
#include "toy/Lexer.h"
#include "toy/MLIRGen.h"

using namespace mlir::toy;

namespace {
/// Implementation of a simple MLIR emission fom the Toy AST.
///
/// This will emit operations that are specific to the Toy language, preserving
/// the semantics of the language and (hopefully) allow to perform accurate
/// analysis and transformation based on these high level semantics.
class MLIRGenImpl {
public:
  MLIRGenImpl(mlir::MLIRContext &context) : builder(&context) {}

  /// Public API: convert the AST for a Toy module (source file) to an MLIR
  /// Module operation.
  mlir::ModuleOp mlirGen(toy::ModuleAST &moduleAST) {
    // We create an empty MLIR module and codegen functions one at a time and
    // add them to the module.
    module = mlir::ModuleOp::create(builder.getUnknownLoc());

    for (toy::FunctionAST &f : moduleAST) {
      mlirGen(f);
    }

    // Verify the module after we have finished constructing it, this will check
    // the structural properties of the IR and invoke any specific verifiers we
    // have on the Toy operations.
    if (failed(mlir::verify(module))) {
      module->emitError("module verification error");
      return nullptr;
    }

    return module;
  }

private:
  /// A "module" matches a Toy source file: containing a list of functions.
  mlir::ModuleOp module;

  /// The builder is a helper class to create IR inside a function. The builder
  /// is stateful, in particular it keeps an "insertion point": this is where
  /// the next operation will be introduced.
  mlir::OpBuilder builder;

  /// The symbol table maps a variable name to a valu in the current scope.
  /// Entering a function creates a new scope, and the function arguments are
  /// added to the mapping. When the processing of a function is terminated, the
  /// scope is destroyed and the mappings crated in this scope are dropped.
  llvm::ScopedHashTable<llvm::StringRef, mlir::Value> symbolTable;

  /// Helper conversion for a Toy AST location to an MLIR location.
  mlir::Location loc(const toy::SourceLocation &loc) {
    return mlir::FileLineColLoc::get(builder.getStringAttr(*loc.file),
                                     static_cast<unsigned>(loc.line),
                                     static_cast<unsigned>(loc.col));
  }

  /// Declare a variable in the current scope, return success if the variable
  /// wasn't declared yet.
  llvm::LogicalResult declare(llvm::StringRef var, mlir::Value value) {
    if (symbolTable.count(var)) {
      return mlir::failure();
    }

    symbolTable.insert(var, value);
    return mlir::success();
  }

  /// Build a tensor type from a list of shape dimensions.
  mlir::Type getType(llvm::ArrayRef<int64_t> shape) {
    // If the shape is empty, then this type is unranked.
    if (shape.empty()) {
      return mlir::UnrankedTensorType::get(builder.getF64Type());
    }

    // Otherwise, we use the given shape.
    return mlir::RankedTensorType::get(shape, builder.getF64Type());
  }

  /// Build an MLIR type from a Toy AST variable type (forward to the generic
  /// getType above).
  mlir::Type getType(const toy::VarType &type) { return getType(type.shape); }

  /// Create the prototype for an MLIR function with as many arguments as the
  /// provided Toy AST prototype.
  mlir::toy::FuncOp mlirGen(toy::PrototypeAST &proto) {
    auto location = loc(proto.loc());

    // This is a generic function, the return type will be inferred later.
    // Arguments type are uniformly unranked tensors.
    llvm::SmallVector<mlir::Type, 4> argTypes(proto.getArgs().size(),
                                              getType(toy::VarType().shape));

    auto funcType = builder.getFunctionType(argTypes, {});
    return mlir::toy::FuncOp::create(builder, location, proto.getName(),
                                     funcType);
  }

  /// Emit a new function and add it to the MLIR module
  mlir::toy::FuncOp mlirGen(toy::FunctionAST &funcAST) {
    // Create a scope in the symbol table to hold variable declarations.
    // The idea here is to have a linked list of hash tables that manage the
    // scope of the variable.
    llvm::ScopedHashTableScope<llvm::StringRef, mlir::Value> varScope(
        symbolTable);

    // Create a MLIR function for the given prototype.
    builder.setInsertionPointToEnd(module.getBody());
    mlir::toy::FuncOp function = mlirGen(*funcAST.getProto());
    if (!function) {
      return nullptr;
    }

    // Let's start the body of the function now
    mlir::Block &entryBlock = function.front();
    auto protoArgs = funcAST.getProto()->getArgs();

    // Declare all the function arguments in the symbol table.
    for (const auto nameValue :
         llvm::zip(protoArgs, entryBlock.getArguments())) {
      if (failed(declare(std::get<0>(nameValue)->getName(),
                         std::get<1>(nameValue)))) {
        return nullptr;
      }
    }

    // Set the insertion point in the builder to the beginning of the function
    // body, it will be used throughout the codegen to create operations in this
    // function.
    builder.setInsertionPointToEnd(&entryBlock);

    // Emit the body of the function.
    if (mlir::failed(mlirGen(*funcAST.getBody()))) {
      function.erase();
      return nullptr;
    }

    // Implicitly return void if no return statement was emitted.
    ReturnOp returnOp;
    if (!entryBlock.empty()) {
      returnOp = llvm::dyn_cast<ReturnOp>(entryBlock.back());
    }

    if (!returnOp) {
      ReturnOp::create(builder, loc(funcAST.getProto()->loc()));
    } else if (returnOp.hasOperand()) {
      // Otherwise, if this return operation has an operand then add a result to
      // the function.
      function.setType(
          builder.getFunctionType(function.getFunctionType().getInputs(),
                                  getType(toy::VarType().shape)));
    }

    // If this function isn't main, then set the visibility to private.
    if (funcAST.getProto()->getName() != "main") {
      function.setPrivate();
    }

    return function;
  }

  /// Emit a binary operation
  mlir::Value mlirGen(toy::BinaryExprAST &binop) {
    // First emit the operations for each side of the operation before emitting
    // the operation itself. For example if the expression is `a + foo(a)`
    // 1) First it will visiting the LHS, which will return a reference to the
    //    value holding `a`. This value should have been emitted at declaration
    //    time and registered in the symbol table, so nothing would be
    //    codegen'd. If the value is not in the symbol table, an error has been
    //    emitted and nullptr is returned.
    // 2) Then the RHS is visited (recursively) and a call to `foo` is emitted
    //    and the result value is returned. If an error occurs we get a nullptr
    //    and propagate.
    mlir::Value lhs = mlirGen(*binop.getLhs());
    if (!lhs) {
      return nullptr;
    }

    mlir::Value rhs = mlirGen(*binop.getRhs());
    if (!rhs) {
      return nullptr;
    }

    auto location = loc(binop.loc());

    // Reconcile the shapes of the two operands following NumPy broadcasting
    // rules.
    auto lhsType = llvm::dyn_cast<mlir::RankedTensorType>(lhs.getType());
    auto rhsType = llvm::dyn_cast<mlir::RankedTensorType>(rhs.getType());

    if (lhsType && rhsType && lhsType != rhsType) {
      auto lhsShape = lhsType.getShape();
      auto rhsShape = rhsType.getShape();

      llvm::SmallVector<int64_t, 4> resultShape;
      int i = static_cast<int>(lhsShape.size()) - 1;
      int j = static_cast<int>(rhsShape.size()) - 1;
      bool compatible = true;

      while (i >= 0 || j >= 0) {
        int64_t dim1 = (i >= 0) ? lhsShape[static_cast<size_t>(i)] : 1;
        int64_t dim2 = (j >= 0) ? rhsShape[static_cast<size_t>(j)] : 1;
        if (dim1 == dim2) {
          resultShape.push_back(dim1);
        } else if (dim1 == 1) {
          resultShape.push_back(dim2);
        } else if (dim2 == 1) {
          resultShape.push_back(dim1);
        } else {
          compatible = false;
          break;
        }
        i--;
        j--;
      }

      if (compatible) {
        std::reverse(resultShape.begin(), resultShape.end());
        auto resultType =
            mlir::RankedTensorType::get(resultShape, builder.getF64Type());

        if (lhsType != resultType) {
          lhs = BroadcastOp::create(builder, location, lhs, resultType);
        }

        if (rhsType != resultType) {
          rhs = BroadcastOp::create(builder, location, rhs, resultType);
        }
      }
    }

    // Derive the operation name from the binary operator.
    switch (binop.getOp()) {
    case '+':
      return AddOp::create(builder, location, lhs, rhs);
    case '*':
      return MulOp::create(builder, location, lhs, rhs);
    }

    emitError(location, "invalid binary operator '") << binop.getOp() << "'";
    return nullptr;
  }

  /// This is a reference to a variable in a expression. The variable is
  /// expected to have been declared and sho should have a value in the symbol
  /// table, otherwise emit an error and return nullptr.
  mlir::Value mlirGen(toy::VariableExprAST &expr) {
    if (auto variable = symbolTable.lookup(expr.getName())) {
      return variable;
    }

    mlir::emitError(loc(expr.loc()), "error: unknown variable '")
        << expr.getName() << "'";
    return nullptr;
  }

  /// Emit a return operation. This will return failure if any generation fails.
  llvm::LogicalResult mlirGen(toy::ReturnExprAST &ret) {
    auto location = loc(ret.loc());

    // 'return' takes a optional expression, handles that case here.
    mlir::Value expr = nullptr;
    if (ret.getExpr().has_value()) {
      if (!(expr = mlirGen(**ret.getExpr()))) {
        return mlir::failure();
      }
    }

    // Otherwise, this return operation has zero operands.
    ReturnOp::create(builder, location,
                     expr ? llvm::ArrayRef(expr)
                          : llvm::ArrayRef<mlir::Value>());
    return mlir::success();
  }

  /// Recursive helper function to accumulate the data that compose an array
  /// literal. It flattens the nested structure in the supplied vector. For
  /// example with this array:
  ///  [[1, 2], [3, 4]]
  /// we will generate:
  ///  [ 1, 2, 3, 4 ]
  /// Individual number are represented as doubles.
  /// Attributes are the way MLIR attaches constant to operations.
  void collectData(toy::ExprAST &expr, std::vector<double> &data) {
    if (auto *lit = llvm::dyn_cast<toy::LiteralExprAST>(&expr)) {
      for (auto &value : lit->getValues()) {
        collectData(*value, data);
      }

      return;
    }

    assert(llvm::isa<toy::NumberExprAST>(expr) &&
           "expected literal or number expr");
    data.push_back(llvm::cast<toy::NumberExprAST>(expr).getValue());
  }

  /// Emit a literal/constant array. It will be emitted as a flattened array of
  /// data in a Attribute attached to a `toy.constant` operation.
  ///
  /// From the documentation:
  /// "Attributes are the mechanism for specifying constant data in MLIR in
  /// places where a variable is never allowed [...]. They consist of a name and
  /// a concrete attribute value. The set of expected attributes, their
  /// structure, and their interpretation are all contextually dependent on what
  /// they are attached to."
  ///
  /// Example the source level statement:
  ///  var a<2, 3> = [[1, 2, 3], [4, 5, 6]];
  /// will be converted to:
  ///  %0 = "toy.constant"() {value: dense<tensor<2x3xf64>,
  ///     [[1.000000e+00, 2.000000e+00, 3.000000e+00],
  ///      [4.000000e+00, 5.000000e+00, 6.000000e+00]]>} : () -> tensor<2x3xf64>
  mlir::Value mlirGen(toy::LiteralExprAST &lit) {
    auto type = getType(lit.getDims());

    // The attribute is a vector with a floatin point value per element (number)
    // in the array.
    std::vector<double> data;
    data.reserve(std::accumulate(lit.getDims().begin(), lit.getDims().end(),
                                 size_t(1), std::multiplies<size_t>()));
    collectData(lit, data);

    // The type of this attribute is a tensor of 64-bit floating-point with the
    // shape of the literal.
    mlir::Type elementType = builder.getF64Type();
    auto dataType = mlir::RankedTensorType::get(lit.getDims(), elementType);

    // This is the actual attribute that holds the list of values for this
    // tensor literal.
    auto dataAttribute =
        mlir::DenseElementsAttr::get(dataType, llvm::ArrayRef(data));

    // Build the MLIR op `toy.constant`. This invokes the `ConstantOp::build`
    // method.
    return ConstantOp::create(builder, loc(lit.loc()), type, dataAttribute);
  }

  /// Emit a call expression. It emits specific operations for the `tranpose`
  /// builtin. Other identifiers are assumed to be user-defined functions.
  mlir::Value mlirGen(toy::CallExprAST &call) {
    llvm::StringRef callee = call.getCallee();
    auto location = loc(call.loc());

    // Codegen the operands first.
    mlir::SmallVector<mlir::Value, 4> operands;
    for (auto &expr : call.getArgs()) {
      auto arg = mlirGen(*expr);
      if (!arg) {
        return nullptr;
      }

      operands.push_back(arg);
    }

    // Calls to user-defined functions are mapped to a custom call that takes
    // the callee name as an attribute.
    return GenericCallOp::create(builder, location, callee, operands);
  }

  /// Emit a call expression to the builtin 'transpose' function.
  mlir::Value mlirGen(toy::TransposeExprAST &call) {
    auto location = loc(call.loc());

    // Codegen the operand first.
    auto arg = mlirGen(*call.getArg());
    if (!arg) {
      return nullptr;
    }

    return TransposeOp::create(builder, location, arg);
  }

  /// Emit a call expression to the builtin 'print' function.
  llvm::LogicalResult mlirGen(toy::PrintExprAST &call) {
    auto location = loc(call.loc());

    // Codegen the operand first.
    auto arg = mlirGen(*call.getArg());
    if (!arg) {
      return mlir::failure();
    }

    PrintOp::create(builder, location, arg);
    return mlir::success();
  }

  /// Emit a constant for a single number.
  mlir::Value mlirGen(toy::NumberExprAST &num) {
    return ConstantOp::create(builder, loc(num.loc()), num.getValue());
  }

  /// Dispatch codegen for the right expression subclass using RTTI.
  mlir::Value mlirGen(toy::ExprAST &expr) {
    switch (expr.getKind()) {
    case toy::ExprAST::ExprASTKind::BinOp:
      return mlirGen(llvm::cast<toy::BinaryExprAST>(expr));
    case toy::ExprAST::ExprASTKind::Var:
      return mlirGen(llvm::cast<toy::VariableExprAST>(expr));
    case toy::ExprAST::ExprASTKind::Literal:
      return mlirGen(llvm::cast<toy::LiteralExprAST>(expr));
    case toy::ExprAST::ExprASTKind::Call:
      return mlirGen(llvm::cast<toy::CallExprAST>(expr));
    case toy::ExprAST::ExprASTKind::Transpose:
      return mlirGen(llvm::cast<toy::TransposeExprAST>(expr));
    case toy::ExprAST::ExprASTKind::Num:
      return mlirGen(llvm::cast<toy::NumberExprAST>(expr));
    case toy::ExprAST::ExprASTKind::VarDecl:
      return mlirGen(llvm::cast<toy::VarDeclExprAST>(expr));
    default:
      emitError(loc(expr.loc()))
          << "MLIR codegen encountered an unhandled expr kind '"
          << llvm::Twine(expr.getStrKind()) << "'";
      return nullptr;
    }
  }

  /// Handle a variable declaration, we'll codegen the expression that forms the
  /// initializer and record the value in the symbol table before returning it.
  /// Future expressions will be able to reference this variable through symbol
  /// table lookup.
  mlir::Value mlirGen(toy::VarDeclExprAST &vardecl) {
    auto *init = vardecl.getInitVal();
    if (!init) {
      mlir::emitError(loc(vardecl.loc()),
                      "missing initializer in variable declaration");
      return nullptr;
    }

    mlir::Value value = mlirGen(*init);
    if (!value) {
      return nullptr;
    }

    // We have the initializer value, but in case the variable was declared with
    // specific shape, we emit a "reshape" operation. It will get optimized out
    // later as needed.
    if (!vardecl.getType().shape.empty()) {
      value = ReshapeOp::create(builder, loc(vardecl.loc()),
                                getType(vardecl.getType()), value);
    }

    // Register the value in the symbol table.
    if (failed(declare(vardecl.getName(), value))) {
      return nullptr;
    }
    return value;
  }

  /// Codegen a list of expressions, return failure if one of them hit an error.
  llvm::LogicalResult mlirGen(toy::ExprASTList &blockAST) {
    llvm::ScopedHashTableScope<llvm::StringRef, mlir::Value> varScope(
        symbolTable);
    for (auto &expr : blockAST) {
      if (auto *ret = llvm::dyn_cast<toy::ReturnExprAST>(expr.get())) {
        return mlirGen(*ret);
      }

      if (auto *print = llvm::dyn_cast<toy::PrintExprAST>(expr.get())) {
        if (mlir::failed(mlirGen(*print))) {
          return mlir::failure();
        }
        continue;
      }

      auto value = mlirGen(*expr);
      if (!value) {
        return mlir::failure();
      }
    }
    return mlir::success();
  }
};

} // namespace

namespace toy {

// The public API for codegen
mlir::OwningOpRef<mlir::ModuleOp> mlirGen(mlir::MLIRContext &context,
                                          ModuleAST &moduleAST) {
  return MLIRGenImpl(context).mlirGen(moduleAST);
}

} // namespace toy
