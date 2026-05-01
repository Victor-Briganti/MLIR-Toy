// RUN: toy-compiler --opt -x=mlir %s --emit=mlir-llvm 2>&1 | FileCheck %s

toy.func @main() {
  // CHECK: %[[ALLOC:.*]] = llvm.call @malloc
  // CHECK: %[[STRUCT:.*]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: llvm.insertvalue %[[ALLOC]], %[[STRUCT]][0]
  // CHECK: llvm.insertvalue %[[ALLOC]], %{{.*}}[1]
  %0 = toy.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>

  // CHECK: llvm.store %{{.*}}, %{{.*}} : f64, !llvm.ptr
  // CHECK: llvm.store %{{.*}}, %{{.*}} : f64, !llvm.ptr
  // CHECK: llvm.store %{{.*}}, %{{.*}} : f64, !llvm.ptr
  // CHECK: llvm.store %{{.*}}, %{{.*}} : f64, !llvm.ptr
  // CHECK: llvm.store %{{.*}}, %{{.*}} : f64, !llvm.ptr
  // CHECK: llvm.store %{{.*}}, %{{.*}} : f64, !llvm.ptr
  toy.print %0 : tensor<2x3xf64>
  // CHECK: llvm.call @free(%{{.*}}) : (!llvm.ptr) -> ()
  // CHECK: llvm.return
  toy.return
}
