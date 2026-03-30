// RUN: iree-opt --split-input-file --mlir-print-local-scope --iree-preprocessing-convert-strided-insert-slice-to-generic %s | FileCheck %s

// No transformation: passthrough element count (32*2048=65536) exceeds threshold.
util.func public @no_transform_large_passthrough(%src: tensor<32x25x25x2048xf16>) -> tensor<32x50x50x2048xf16> {
  %cst = arith.constant dense<0.000000e+00> : tensor<32x50x50x2048xf16>
  %0 = tensor.insert_slice %src into %cst[0, 0, 0, 0] [32, 25, 25, 2048] [1, 2, 2, 1] : tensor<32x25x25x2048xf16> into tensor<32x50x50x2048xf16>
  util.return %0 : tensor<32x50x50x2048xf16>
}

// CHECK-LABEL: @no_transform_large_passthrough
// CHECK:       tensor.insert_slice
// CHECK-NOT:   linalg.generic

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

// -----

// Converted: small passthrough count (no passthrough dims), stride-2.
util.func public @stride2_no_passthrough(%src: tensor<4x4xf16>) -> tensor<9x9xf16> {
  %cst = arith.constant dense<0.000000e+00> : tensor<9x9xf16>
  %0 = tensor.insert_slice %src into %cst[1, 1] [4, 4] [2, 2] : tensor<4x4xf16> into tensor<9x9xf16>
  util.return %0 : tensor<9x9xf16>
}

// CHECK-LABEL: @stride2_no_passthrough
// CHECK-NOT:   tensor.insert_slice
// CHECK:       linalg.generic

// -----

// Converted with dim collapse: passthrough product (1*32=32) under threshold.
// Trailing passthrough dims are collapsed from 2D to 1D.
util.func public @stride2_small_passthrough_collapse(%src: tensor<1x25x25x4x8xf16>) -> tensor<1x52x52x4x8xf16> {
  %cst = arith.constant dense<0.000000e+00> : tensor<1x52x52x4x8xf16>
  %0 = tensor.insert_slice %src into %cst[0, 1, 1, 0, 0] [1, 25, 25, 4, 8] [1, 2, 2, 1, 1] : tensor<1x25x25x4x8xf16> into tensor<1x52x52x4x8xf16>
  util.return %0 : tensor<1x52x52x4x8xf16>
}

// CHECK-LABEL: @stride2_small_passthrough_collapse
// CHECK-SAME:      %[[SRC:.*]]: tensor<1x25x25x4x8xf16>
// CHECK-NOT:   tensor.insert_slice
// CHECK:       %[[CSRC:.*]] = tensor.collapse_shape %[[SRC]]
// CHECK-SAME:      tensor<1x25x25x4x8xf16> into tensor<1x25x25x32xf16>
// CHECK:       %[[GENERIC:.*]] = linalg.generic
// CHECK-SAME:      iterator_types = ["parallel", "parallel", "parallel", "parallel"]
// CHECK:         tensor.extract %[[CSRC]]
// CHECK:         arith.select
// CHECK:       %[[EXP:.*]] = tensor.expand_shape %[[GENERIC]]
// CHECK-SAME:      tensor<1x52x52x32xf16> into tensor<1x52x52x4x8xf16>
// CHECK:       util.return %[[EXP]]

// -----

// No transformation: non-batch passthrough product (8*56=448) exceeds threshold.
// Grouped backward-data convolution where DMA is more efficient.
util.func public @no_transform_grouped_conv(%src: tensor<2x118x182x8x56xbf16>) -> tensor<2x237x365x8x56xbf16> {
  %cst = arith.constant dense<0.000000e+00> : tensor<2x237x365x8x56xbf16>
  %0 = tensor.insert_slice %src into %cst[0, 1, 1, 0, 0] [2, 118, 182, 8, 56] [1, 2, 2, 1, 1] : tensor<2x118x182x8x56xbf16> into tensor<2x237x365x8x56xbf16>
  util.return %0 : tensor<2x237x365x8x56xbf16>
}

// CHECK-LABEL: @no_transform_grouped_conv
// CHECK:       tensor.insert_slice
// CHECK-NOT:   linalg.generic
