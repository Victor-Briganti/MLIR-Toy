// RUN: toy-compiler --opt -x=mlir %s --emit=mlir-affine 2>&1 | FileCheck %s

// CHECK-LABEL: func.func @main()
toy.func @main() {
    // CHECK: %[[CST0:.*]] = arith.constant 6.000000e+00 : f64
  // CHECK: %[[CST1:.*]] = arith.constant 5.000000e+00 : f64
  // CHECK: %[[CST2:.*]] = arith.constant 4.000000e+00 : f64
  // CHECK: %[[CST3:.*]] = arith.constant 3.000000e+00 : f64
  // CHECK: %[[CST4:.*]] = arith.constant 2.000000e+00 : f64
  // CHECK: %[[CST5:.*]] = arith.constant 1.000000e+00 : f64
  // CHECK: %[[ALLOC0:.*]] = memref.alloc() : memref<3x2xf64>
  // CHECK: %[[ALLOC1:.*]] = memref.alloc() : memref<2x3xf64>
  // CHECK: affine.store %[[CST5]], %[[ALLOC1]][0, 0] : memref<2x3xf64>
  // CHECK: affine.store %[[CST4]], %[[ALLOC1]][0, 1] : memref<2x3xf64>
  // CHECK: affine.store %[[CST3]], %[[ALLOC1]][0, 2] : memref<2x3xf64>
  // CHECK: affine.store %[[CST2]], %[[ALLOC1]][1, 0] : memref<2x3xf64>
  // CHECK: affine.store %[[CST1]], %[[ALLOC1]][1, 1] : memref<2x3xf64>
  // CHECK: affine.store %[[CST0]], %[[ALLOC1]][1, 2] : memref<2x3xf64>
  %0 = toy.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>
  %2 = toy.transpose(%0 : tensor<2x3xf64>) to tensor<3x2xf64>
  // CHECK: affine.for %[[ARG0:.*]] = 0 to 3
  // CHECK: affine.for %[[ARG1:.*]] = 0 to 2 
  // CHECK: %[[CONST0:.*]] = affine.load %[[ALLOC1]][%[[ARG1]], %[[ARG0]]] : memref<2x3xf64>
  // CHECK: %[[CONST1:.*]] = arith.mulf %[[CONST0]], %[[CONST0]] : f64
  // CHECK: affine.store %[[CONST1]], %[[ALLOC0]][%[[ARG0]], %[[ARG1]]] : memref<3x2xf64>
  %3 = toy.mul %2, %2 : tensor<3x2xf64>
  // CHECK: toy.print %[[ALLOC0]] : memref<3x2xf64>
  toy.print %3 : tensor<3x2xf64>
  // CHECK: memref.dealloc %[[ALLOC1]] : memref<2x3xf64>
  // CHECK: memref.dealloc %[[ALLOC0]] : memref<3x2xf64>
  // CHECK: return
  toy.return
}