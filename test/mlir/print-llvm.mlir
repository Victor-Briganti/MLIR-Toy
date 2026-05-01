// RUN: toy-compiler --opt -x=mlir %s --emit=mlir-llvm 2>&1 | FileCheck %s

// CHECK: llvm.func @printf(!llvm.ptr, ...) -> i32

toy.func @main() {
  %0 = toy.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf64>

  // CHECK: llvm.br ^bb1
  // CHECK: ^bb1(%[[IV1:.*]]: i64):
  // CHECK:   llvm.cond_br %{{.*}}, ^bb2, ^bb6
  // CHECK: ^bb2:
  // CHECK:   llvm.br ^bb3
  // CHECK: ^bb3(%[[IV2:.*]]: i64):
  // CHECK:   llvm.cond_br %{{.*}}, ^bb4, ^bb5
  // CHECK: ^bb4:
  // CHECK:   %[[VAL:.*]] = llvm.load %{{.*}} : !llvm.ptr -> f64
  // CHECK:   llvm.call @printf(%{{.*}}, %[[VAL]])
  // CHECK:   llvm.br ^bb3
  // CHECK: ^bb5:
  // CHECK:   llvm.call @printf(%{{.*}})
  // CHECK:   llvm.br ^bb1
  toy.print %0 : tensor<2x3xf64>
  toy.return
}
