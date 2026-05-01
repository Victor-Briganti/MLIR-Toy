// RUN: toy-compiler -x=mlir %s --emit=mlir-affine 2>&1 | FileCheck %s

// CHECK-LABEL: func.func @main()
toy.func @main() {
  // CHECK: %[[CST0:.*]] = arith.constant 6.000000e+00 : f64
  // CHECK: %[[CST1:.*]] = arith.constant 5.000000e+00 : f64
  // CHECK: %[[CST2:.*]] = arith.constant 4.000000e+00 : f64
  // CHECK: %[[CST3:.*]] = arith.constant 3.000000e+00 : f64
  // CHECK: %[[CST4:.*]] = arith.constant 2.000000e+00 : f64
  // CHECK: %[[CST5:.*]] = arith.constant 1.000000e+00 : f64
  // CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<2x3xf64>
  // CHECK: affine.store %[[CST5]], %[[ALLOC]][0, 0] : memref<2x3xf64>
  // CHECK: affine.store %[[CST4]], %[[ALLOC]][0, 1] : memref<2x3xf64>
  // CHECK: affine.store %[[CST3]], %[[ALLOC]][0, 2] : memref<2x3xf64>
  // CHECK: affine.store %[[CST2]], %[[ALLOC]][1, 0] : memref<2x3xf64>
  // CHECK: affine.store %[[CST1]], %[[ALLOC]][1, 1] : memref<2x3xf64>
  // CHECK: affine.store %[[CST0]], %[[ALLOC]][1, 2] : memref<2x3xf64>
  %0 = toy.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>
  // CHECK: toy.print %[[ALLOC]] : memref<2x3xf64>
  toy.print %0 : tensor<2x3xf64>
  // CHECK: memref.dealloc %[[ALLOC]] : memref<2x3xf64>
  // CHECK: return
  toy.return
}