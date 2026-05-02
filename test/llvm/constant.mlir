// RUN: toy-compiler --opt -x=mlir %s --emit=llvm 2>&1 | FileCheck %s

// CHECK-LABEL: define void @main()
toy.func @main() {
  // CHECK: %[[ALLOC0:.*]] = tail call dereferenceable_or_null(48) ptr @malloc(i64 48)
  // CHECK-NEXT: store double 1.000000e+00, ptr %[[ALLOC0]]
  // CHECK-NEXT: getelementptr inbounds nuw i8, ptr %[[ALLOC0]]
  // CHECK-NEXT: store double 4.000000e+00
  // CHECK-NEXT: getelementptr inbounds nuw i8, ptr %[[ALLOC0]]
  // CHECK-NEXT: store double 2.000000e+00
  // CHECK-NEXT: getelementptr inbounds nuw i8, ptr %[[ALLOC0]]
  // CHECK-NEXT: store double 5.000000e+00
  // CHECK-NEXT: getelementptr inbounds nuw i8, ptr %[[ALLOC0]]
  // CHECK-NEXT: store double 3.000000e+00
  // CHECK-NEXT: getelementptr inbounds nuw i8, ptr %[[ALLOC0]]
  // CHECK-NEXT: store double 6.000000e+00
  // CHECK-NEXT: br label %.preheader, !dbg !10
  %0 = toy.constant dense<[[1.000000e+00, 4.000000e+00], [2.000000e+00, 5.000000e+00], [3.000000e+00, 6.000000e+00]]> : tensor<3x2xf64>
  // CHECK: tail call i32 ({{.*}}) @printf({{.*}})
  // CHECK: tail call i32 @putchar(i32 10)
  toy.print %0 : tensor<3x2xf64>
  // CHECK: tail call void @free({{.*}})
  toy.return
}
