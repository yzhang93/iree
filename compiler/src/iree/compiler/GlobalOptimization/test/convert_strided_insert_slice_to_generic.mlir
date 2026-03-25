// RUN: iree-opt --split-input-file --mlir-print-local-scope --iree-global-opt-convert-strided-insert-slice-to-generic %s | FileCheck %s

// Strided insert_slice into zeros with stride 2 and zero offset.
util.func public @stride2_zero_offset(%src: tensor<32x25x25x2048xf16>) -> tensor<32x50x50x2048xf16> {
  %cst = arith.constant dense<0.000000e+00> : tensor<32x50x50x2048xf16>
  %0 = tensor.insert_slice %src into %cst[0, 0, 0, 0] [32, 25, 25, 2048] [1, 2, 2, 1] : tensor<32x25x25x2048xf16> into tensor<32x50x50x2048xf16>
  util.return %0 : tensor<32x50x50x2048xf16>
}

// CHECK-LABEL: @stride2_zero_offset
// CHECK-SAME:      %[[SRC:.*]]: tensor<32x25x25x2048xf16>
// CHECK-NOT:   tensor.insert_slice
// CHECK:       %[[GENERIC:.*]] = linalg.generic
// CHECK-SAME:      iterator_types = ["parallel", "parallel", "parallel", "parallel"]
// CHECK:         linalg.index 1
// CHECK:         arith.remsi
// CHECK:         arith.divsi
// CHECK:         linalg.index 2
// CHECK:         arith.remsi
// CHECK:         arith.divsi
// CHECK:         scf.if
// CHECK:           tensor.extract %[[SRC]]
// CHECK:           scf.yield
// CHECK:         else
// CHECK:           scf.yield
// CHECK:         linalg.yield
// CHECK:       util.return %[[GENERIC]]

// -----

// Strided insert_slice into zeros with stride 2 and non-zero offset.
util.func public @stride2_nonzero_offset(%src: tensor<32x25x25x32x32xf16>) -> tensor<32x52x52x32x32xf16> {
  %cst = arith.constant dense<0.000000e+00> : tensor<32x52x52x32x32xf16>
  %0 = tensor.insert_slice %src into %cst[0, 1, 1, 0, 0] [32, 25, 25, 32, 32] [1, 2, 2, 1, 1] : tensor<32x25x25x32x32xf16> into tensor<32x52x52x32x32xf16>
  util.return %0 : tensor<32x52x52x32x32xf16>
}

// CHECK-LABEL: @stride2_nonzero_offset
// CHECK-SAME:      %[[SRC:.*]]: tensor<32x25x25x32x32xf16>
// CHECK-NOT:   tensor.insert_slice
// CHECK:       %[[GENERIC:.*]] = linalg.generic
// CHECK-SAME:      iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]
// CHECK:         linalg.index 1
// CHECK:         arith.remsi
// CHECK:         arith.divsi
// CHECK:         linalg.index 2
// CHECK:         arith.remsi
// CHECK:         arith.divsi
// CHECK:         scf.if
// CHECK:           tensor.extract %[[SRC]]
// CHECK:           scf.yield
// CHECK:         else
// CHECK:           scf.yield
// CHECK:         linalg.yield
// CHECK:       util.return %[[GENERIC]]

// -----

// Integer element type.
util.func public @stride2_i32(%src: tensor<4x4xi32>) -> tensor<8x8xi32> {
  %cst = arith.constant dense<0> : tensor<8x8xi32>
  %0 = tensor.insert_slice %src into %cst[0, 0] [4, 4] [2, 2] : tensor<4x4xi32> into tensor<8x8xi32>
  util.return %0 : tensor<8x8xi32>
}

// CHECK-LABEL: @stride2_i32
// CHECK-NOT:   tensor.insert_slice
// CHECK:       linalg.generic
// CHECK:         arith.remsi
// CHECK:         scf.if
// CHECK:           tensor.extract

// -----

// No transformation: all strides are 1.
util.func public @no_transform_unit_strides(%src: tensor<32x25x25x2048xf16>) -> tensor<32x50x50x2048xf16> {
  %cst = arith.constant dense<0.000000e+00> : tensor<32x50x50x2048xf16>
  %0 = tensor.insert_slice %src into %cst[0, 1, 1, 0] [32, 25, 25, 2048] [1, 1, 1, 1] : tensor<32x25x25x2048xf16> into tensor<32x50x50x2048xf16>
  util.return %0 : tensor<32x50x50x2048xf16>
}

// CHECK-LABEL: @no_transform_unit_strides
// CHECK:       tensor.insert_slice
// CHECK-NOT:   linalg.generic

// -----

// No transformation: destination is not a zero constant.
util.func public @no_transform_nonzero_dest(%src: tensor<4x4xf32>, %dest: tensor<8x8xf32>) -> tensor<8x8xf32> {
  %0 = tensor.insert_slice %src into %dest[0, 0] [4, 4] [2, 2] : tensor<4x4xf32> into tensor<8x8xf32>
  util.return %0 : tensor<8x8xf32>
}

// CHECK-LABEL: @no_transform_nonzero_dest
// CHECK:       tensor.insert_slice
// CHECK-NOT:   linalg.generic
