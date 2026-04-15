//===--- Ops.h - Toy dialect operation definitions ---------*- tablegen -*-===//
//
// Defines the operations of the toy dialect.
//
//===----------------------------------------------------------------------===//

#ifndef TOY_OPS
#define TOY_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// Provide a definition of the 'toy' dialect in the ODS framework so that we
// can define our operations.
def Toy_Dialect : Dialect {
  let name = "toy";
  let cppNamespace = "::mlir::toy";
}

// Base class for toy dialect operations. This operation inherits from the
// base `Op` class in OpBase.td, and provides:
// * The parent dialect of the operation.
// * The mnemonic for the opretaion, or the name without the dialect prefix.
// * A list of traits for the operation.
class Toy_Op<string mnemonic, list<Trait> traits = []> :
Op<Toy_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

// We define a toy operation by inheriting from our base 'Toy_Op' class above.
// Here we provide the mnemonic and a list of traits for the operation. The
// constant operation is marked as 'Pure' as it is a pure operation
// and may be removed if dead.
def ConstantOp : Toy_Op<"constant", [Pure]> {
  // Provide a summary and a description for this operation. This can be used to
  // auto-generate documentation of the operations within our dialect.
  let summary = "constant";
  let description = [{
    Constant operation turns a literal into a SSA value. THe data is attached
      to the operation as an attribute. For example:

      ```mlir
      %0 = toy.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]>
      : tensor<2x3xf64>
      ```
  }];

  // The cosntant operation takes an attribute as the only input.
  let arguments = (ins F64ElementsAttr:$value);

  // The constant operation returns a single value of TensorType
  let results = (outs F64Tensor);

  // Indicate that the operation has a custom parser and printer method
  let hasCustomAssemblyFormat = 1;

  // Add custom build methods for the constant operation. These method
  // populates the `state` that MLIR uses to create operations, i.e. these
  // are used when using `ConstantOp::create(builder, ...)`.
  let builders = [
    // Build a constant with a given cosntant tensor value.
    OpBuilder<(ins "DenseElementsAttr":$value), [{
      build($_builder, $_state, value.getType(), value);
    }]>,

    // Build a constant with a given constant floatin-point value.
    OpBuilder<(ins "double":$value)>
  ];

  // Indicate that additional verification for this operation is
  // necessary.
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// AddOp
//===----------------------------------------------------------------------===//

def AddOp : Toy_Op<"add"> {
  let summary = "element-wise addition operaiton";
  let description = [{
    The "add" operation performs element-wise addition between two tensors.
      The shapes of the tensor operands are expected to match.
  }];

  let arguments = (ins F64Tensor:$lhs, F64Tensor:$rhs);
  let results = (outs F64Tensor);

  // Indicate that the operation has a custom parser and printer method.
  let hasCustomAssemblyFormat = 1;

  // Allow building an AddOp from the two input operands.
  let builders = [
    OpBuilder<(ins "Value":$lhs, "Value":$rhs)>
  ];
}

//===----------------------------------------------------------------------===//
// FuncOp
//===----------------------------------------------------------------------===//

def FuncOp : Toy_Op<"func", [FunctionOpInterface, IsolatedFromAbove]> {
  let summary = "user defined function operation";
  let description = [{
    The "toy.func" operation represents a user defined function. These are
      callable SSA.region operations that contain toy computations.

      Example:

      ```mlir
      toy.func @main() {
        %0 = toy.constant dense<5.500000e+00> : tensor<f64>
          %1 = toy.reshape(%0 : tensor<f64>) to tensor<2x2xf64>
          toy.print %1 : tensor<2x2xf64>
          toy.return
      }
    ```
  }];

  let arguments = (ins 
      SymbolNameAttr:$sym_name,
      TypeAttrOf<FunctionType>:$function_type,
      OptionalAttr<DictArrayAttr>:$arg_attrs,
      OptionalAttr<DictArrayAttr>:$res_attrs
      );
  let regions = (region AnyRegion:$body);

  let builders = [OpBuilder<(ins 
      "StringRef":$name, "FunctionType":$type,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)
    >];

  let extraClassDeclaration = [{
    //===----------------------------------------------------------------------===//
    // FuncOp
    //===----------------------------------------------------------------------===//

    /// Returns the argument types of this function
    ArrayRef<Ref> getArgumentTypes() { return getFunctionType().getInputs(); }

    /// Returns the result types of this function
    ArrayRef<Ref> getResultTypes() { return getFunctionType().getResults(); }

    Region *getCallableRegion() { return &getBody(); }
  }];

  let hasCustomAssemblyFormat = 1;
  let skipDefaultBuilders = 1;
}

//===----------------------------------------------------------------------===//
// GenericCallOp
//===----------------------------------------------------------------------===//

def GenericCallOp : Toy_Op<"generic_call"> {
  let summary = "generic_call operation";
  let description = [{
    Generic calls represent calls to a user defined function that needs to be
      speciallized for the shape of its arguments. The callee name is attached
      as a symbol reference via an attribute. The arguments list must match the
      arguments expected by the callee. For example:

      ```mlir
      %4 = toy.generic_call @my_func(%1, %3)
      : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64>
         ```

         This is only valid if a function named "my_func" exist and takes two
           arguments.
  }];

  // The generic call operation takes a symbol reference attribute as the
  // callee, and inputs for the call.
  let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<F64Tensor>:$inputs);

  // The generic call operation returns a single value of TensorType.
  let results = (outs F64Tensor);

  // Specialize assembly printing and parsing using a declarative format.
  let assemblyFormat = [{
    $callee `(` $inputs `)` attr-dict `:` functional-type($inputs, results)
  }];


  // Add custom build methos for the generic call operation.
  let builders = [
    OpBuilder<(ins "StringRef":$callee, "ArrayRef<Value>":$arguments)>
  ];
}

//===----------------------------------------------------------------------===//
// MulOP
//===----------------------------------------------------------------------===//

def MulOp : Toy_Op<"mul"> {
  let summary = "element-wise multiplication operation";
  let description = [{
    The "mul" operation performs element-wise multplication between two
      tensors. The shape of the tensor operands are expected to match.
  }];


  let arguments = (ins F64Tensor:$lhs, F64Tensor:$rhs);
  let results = (outs F64Tensor);

  // Indicate that the operation has a custom parser and printer method
  let hasCustomAssemblyFormat = 1;

  // Allow buildin a MulOp with from the two input operands.
  let builders = [
    OpBuilder<(ins "Value":$lhs, "Value":$rhs)>
  ];
}

//===----------------------------------------------------------------------===//
// MulOP
//===----------------------------------------------------------------------===//


#endif // TOY_OPS
