// RUN: toy-compiler -x=mlir %s --emit=mlir-affine 2>&1 | FileCheck %s

// CHECK-LABEL: func.func @main()
toy.func @main() {
  // CHECK: %[[ALLOC0:.*]] = memref.alloc() : memref<3x2xf64>
  // CHECK: %[[ALLOC1:.*]] = memref.alloc() : memref<2x3xf64>
  %0 = toy.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>
  // CHECK: affine.for %[[ARG0:.*]] = 0 to 3 
  // CHECK: affine.for %[[ARG1:.*]] = 0 to 2 
  // CHECK: %[[CONST0:.*]] = affine.load %[[ALLOC1]][%[[ARG1]], %[[ARG0]]] : memref<2x3xf64>
  // CHECK: affine.store %[[CONST0]], %[[ALLOC0]][%[[ARG0]], %[[ARG1]]] : memref<3x2xf64>
  %2 = toy.transpose(%0 : tensor<2x3xf64>) to tensor<3x2xf64>
  toy.print %2 : tensor<3x2xf64>
  // CHECK: memref.dealloc %[[ALLOC1:.*]] : memref<2x3xf64>
  // CHECK: memref.dealloc %[[ALLOC0:.*]] : memref<3x2xf64>
  toy.return
}