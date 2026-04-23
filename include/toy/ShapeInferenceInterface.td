//===- ShapeInferenceInterface.td - Shape Inference Interface -*- tablegen ===//
//
// Part of the MLIR Toy project, under the Apache License v2.0 with LLVM 
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Modified by: Victor Briganti in 2026
//
//===----------------------------------------------------------------------===//
//
// Defines the operations of the Shape Inference Op Interface
//
//===----------------------------------------------------------------------===//

#ifndef TOY_SHAPE_INFERENCE_INTERFACE_OPS
#define TOY_SHAPE_INFERENCE_INTERFACE_OPS

include "mlir/IR/OpBase.td"

def ShapeInferenceOpInterface : OpInterface<"ShapeInference"> {
    let description = [{
        Interface to access a registered method to infer the return types for an
        operation that can be used during type inference.
    }];

    let methods = [
    InterfaceMethod<"Infer and set the output shape for the current operation.",
                    "void", "inferShape">
    ];
}

#endif // TOY_SHAPE_INFERENCE_INTERFACE_OPS