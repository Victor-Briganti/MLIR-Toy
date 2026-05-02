// RUN: toy-compiler --opt -x=mlir %s --emit=llvm 2>&1 | FileCheck %s

// CHECK-LABEL: define void @main()
toy.func @main() {
  // CHECK: %[[ALLOC0:.*]] = tail call dereferenceable_or_null(48) ptr @malloc({{.*}})
  // CHECK: %[[ALLOC1:.*]] = tail call dereferenceable_or_null(48) ptr @malloc({{.*}})
  %0 = toy.constant dense<[[1.000000e+00, 4.000000e+00], [2.000000e+00, 5.000000e+00], [3.000000e+00, 6.000000e+00]]> : tensor<3x2xf64>
  %1 = toy.mul %0, %0 : tensor<3x2xf64>
  toy.print %1 : tensor<3x2xf64>
  // CHECK: tail call void @free(ptr %[[ALLOC1]])
  // CHECK: tail call void @free(ptr nonnull %[[ALLOC0]])
  // CHECK: ret void
  toy.return
}
