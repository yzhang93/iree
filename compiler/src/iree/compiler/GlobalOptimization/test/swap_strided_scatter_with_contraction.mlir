// RUN: iree-opt --split-input-file --mlir-print-local-scope --iree-global-opt-swap-strided-scatter-with-contraction %s | FileCheck %s

// Basic 1x1 backward conv: insert_slice(src, zeros) -> matmul -> truncf.
// The pass should swap to: matmul(src, filter) -> truncf -> insert_slice.

#map_in  = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 + d5, d2 + d6, d4)>
#map_fil = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d5, d6, d3)>
#map_out = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>
#map_ew  = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
func.func @swap_scatter_with_conv_and_truncf(
    %src: tensor<2x5x5x8xf16>,
    %filter: tensor<8x1x1x4xf16>) -> tensor<2x10x10x4xf16> {
  %cst_zero = arith.constant dense<0.000000e+00> : tensor<2x10x10x8xf16>
  %cst_f32_zero = arith.constant 0.000000e+00 : f32
  // Scatter: 2x5x5x8 -> 2x10x10x8 with stride 2, offset 0.
  %scattered = tensor.insert_slice %src into %cst_zero[0, 0, 0, 0] [2, 5, 5, 8] [1, 2, 2, 1]
    : tensor<2x5x5x8xf16> into tensor<2x10x10x8xf16>
  // Conv (1x1, so d5 and d6 iterate over [0,0]):
  %empty_f32 = tensor.empty() : tensor<2x10x10x4xf32>
  %fill = linalg.fill ins(%cst_f32_zero : f32) outs(%empty_f32 : tensor<2x10x10x4xf32>) -> tensor<2x10x10x4xf32>
  %conv = linalg.generic {
      indexing_maps = [#map_in, #map_fil, #map_out],
      iterator_types = ["parallel", "parallel", "parallel", "parallel",
                        "reduction", "reduction", "reduction"]}
    ins(%scattered, %filter : tensor<2x10x10x8xf16>, tensor<8x1x1x4xf16>)
    outs(%fill : tensor<2x10x10x4xf32>) {
  ^bb0(%in0: f16, %in1: f16, %out: f32):
    %a = arith.extf %in0 : f16 to f32
    %b = arith.extf %in1 : f16 to f32
    %c = arith.mulf %a, %b : f32
    %d = arith.addf %out, %c : f32
    linalg.yield %d : f32
  } -> tensor<2x10x10x4xf32>
  // Truncf consumer:
  %empty_f16 = tensor.empty() : tensor<2x10x10x4xf16>
  %truncf = linalg.generic {
      indexing_maps = [#map_ew, #map_ew],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
    ins(%conv : tensor<2x10x10x4xf32>)
    outs(%empty_f16 : tensor<2x10x10x4xf16>) {
  ^bb0(%in: f32, %out: f16):
    %t = arith.truncf %in : f32 to f16
    linalg.yield %t : f16
  } -> tensor<2x10x10x4xf16>
  return %truncf : tensor<2x10x10x4xf16>
}

// CHECK-LABEL: @swap_scatter_with_conv_and_truncf
// CHECK-SAME:      %[[SRC:.*]]: tensor<2x5x5x8xf16>, %[[FIL:.*]]: tensor<8x1x1x4xf16>
// The small conv now operates directly on the source:
// CHECK:       linalg.generic
// CHECK-SAME:      ins(%[[SRC]], %[[FIL]] :
// CHECK:       -> tensor<2x5x5x4xf32>
// Truncf on the small result:
// CHECK:       linalg.generic
// CHECK:       -> tensor<2x5x5x4xf16>
// Output scatter:
// CHECK:       tensor.insert_slice
// CHECK-SAME:      [0, 0, 0, 0] [2, 5, 5, 4] [1, 2, 2, 1]
// CHECK-SAME:      tensor<2x5x5x4xf16> into tensor<2x10x10x4xf16>

// -----

// No transformation: all strides are 1 (no scatter).
#map_in2  = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map_fil2 = affine_map<(d0, d1, d2, d3) -> (d3, d2)>
#map_out2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
func.func @no_swap_unit_strides(
    %src: tensor<2x10x8xf16>,
    %filter: tensor<8x4xf16>) -> tensor<2x10x4xf32> {
  %cst = arith.constant dense<0.000000e+00> : tensor<2x10x8xf16>
  %inserted = tensor.insert_slice %src into %cst[0, 0, 0] [2, 10, 8] [1, 1, 1]
    : tensor<2x10x8xf16> into tensor<2x10x8xf16>
  %cst_f32 = arith.constant 0.000000e+00 : f32
  %empty = tensor.empty() : tensor<2x10x4xf32>
  %fill = linalg.fill ins(%cst_f32 : f32) outs(%empty : tensor<2x10x4xf32>) -> tensor<2x10x4xf32>
  %result = linalg.generic {
      indexing_maps = [#map_in2, #map_fil2, #map_out2],
      iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
    ins(%inserted, %filter : tensor<2x10x8xf16>, tensor<8x4xf16>)
    outs(%fill : tensor<2x10x4xf32>) {
  ^bb0(%in0: f16, %in1: f16, %out: f32):
    %a = arith.extf %in0 : f16 to f32
    %b = arith.extf %in1 : f16 to f32
    %c = arith.mulf %a, %b : f32
    %d = arith.addf %out, %c : f32
    linalg.yield %d : f32
  } -> tensor<2x10x4xf32>
  return %result : tensor<2x10x4xf32>
}

// CHECK-LABEL: @no_swap_unit_strides
// CHECK:       tensor.insert_slice
// CHECK:       linalg.generic
// CHECK-NOT:   tensor.insert_slice

// -----

// No transformation: destination is not zero.
#map_mm_a = affine_map<(d0, d1, d2) -> (d0, d2)>
#map_mm_b = affine_map<(d0, d1, d2) -> (d2, d1)>
#map_mm_c = affine_map<(d0, d1, d2) -> (d0, d1)>
func.func @no_swap_nonzero_dest(
    %src: tensor<4x4xf16>,
    %dest: tensor<8x4xf16>,
    %filter: tensor<4x2xf16>) -> tensor<8x2xf32> {
  %scattered = tensor.insert_slice %src into %dest[0, 0] [4, 4] [2, 1]
    : tensor<4x4xf16> into tensor<8x4xf16>
  %cst = arith.constant 0.000000e+00 : f32
  %empty = tensor.empty() : tensor<8x2xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<8x2xf32>) -> tensor<8x2xf32>
  %result = linalg.generic {
      indexing_maps = [#map_mm_a, #map_mm_b, #map_mm_c],
      iterator_types = ["parallel", "parallel", "reduction"]}
    ins(%scattered, %filter : tensor<8x4xf16>, tensor<4x2xf16>)
    outs(%fill : tensor<8x2xf32>) {
  ^bb0(%in0: f16, %in1: f16, %out: f32):
    %a = arith.extf %in0 : f16 to f32
    %b = arith.extf %in1 : f16 to f32
    %c = arith.mulf %a, %b : f32
    %d = arith.addf %out, %c : f32
    linalg.yield %d : f32
  } -> tensor<8x2xf32>
  return %result : tensor<8x2xf32>
}

// CHECK-LABEL: @no_swap_nonzero_dest
// CHECK:       tensor.insert_slice
// CHECK:       linalg.generic
// CHECK-NOT:   tensor.insert_slice

// -----

// No transformation: 3x3 conv (non-projected-permutation scatter map).
// The scatter map has d1+d5, d2+d6 which are not single parallel dims for
// strided dims when kernel > 1.
#map_conv_in  = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 + d5, d2 + d6, d4)>
#map_conv_fil = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d5, d6, d3)>
#map_conv_out = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>
func.func @no_swap_3x3_conv(
    %src: tensor<1x5x5x8xf16>,
    %filter: tensor<8x3x3x4xf16>) -> tensor<1x10x10x4xf32> {
  %cst_zero = arith.constant dense<0.000000e+00> : tensor<1x12x12x8xf16>
  %cst_f32_zero = arith.constant 0.000000e+00 : f32
  %scattered = tensor.insert_slice %src into %cst_zero[0, 1, 1, 0] [1, 5, 5, 8] [1, 2, 2, 1]
    : tensor<1x5x5x8xf16> into tensor<1x12x12x8xf16>
  %empty = tensor.empty() : tensor<1x10x10x4xf32>
  %fill = linalg.fill ins(%cst_f32_zero : f32) outs(%empty : tensor<1x10x10x4xf32>) -> tensor<1x10x10x4xf32>
  %conv = linalg.generic {
      indexing_maps = [#map_conv_in, #map_conv_fil, #map_conv_out],
      iterator_types = ["parallel", "parallel", "parallel", "parallel",
                        "reduction", "reduction", "reduction"]}
    ins(%scattered, %filter : tensor<1x12x12x8xf16>, tensor<8x3x3x4xf16>)
    outs(%fill : tensor<1x10x10x4xf32>) {
  ^bb0(%in0: f16, %in1: f16, %out: f32):
    %a = arith.extf %in0 : f16 to f32
    %b = arith.extf %in1 : f16 to f32
    %c = arith.mulf %a, %b : f32
    %d = arith.addf %out, %c : f32
    linalg.yield %d : f32
  } -> tensor<1x10x10x4xf32>
  return %conv : tensor<1x10x10x4xf32>
}

// CHECK-LABEL: @no_swap_3x3_conv
// The scatter + conv remain unchanged for 3x3 (reduction dims > 1).
// CHECK:       tensor.insert_slice
// CHECK:       linalg.generic
// CHECK-NOT:   tensor.insert_slice

// -----

// bf16 test: swap should work for bf16 too (no type restriction).
#map_mm_a2 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map_mm_b2 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map_mm_c2 = affine_map<(d0, d1, d2) -> (d0, d1)>
func.func @swap_bf16_matmul(
    %src: tensor<4x8xbf16>,
    %filter: tensor<8x4xbf16>) -> tensor<10x4xf32> {
  %cst = arith.constant dense<0.000000e+00> : tensor<10x8xbf16>
  %scattered = tensor.insert_slice %src into %cst[1, 0] [4, 8] [2, 1]
    : tensor<4x8xbf16> into tensor<10x8xbf16>
  %cst_f32 = arith.constant 0.000000e+00 : f32
  %empty = tensor.empty() : tensor<10x4xf32>
  %fill = linalg.fill ins(%cst_f32 : f32) outs(%empty : tensor<10x4xf32>) -> tensor<10x4xf32>
  %result = linalg.generic {
      indexing_maps = [#map_mm_a2, #map_mm_b2, #map_mm_c2],
      iterator_types = ["parallel", "parallel", "reduction"]}
    ins(%scattered, %filter : tensor<10x8xbf16>, tensor<8x4xbf16>)
    outs(%fill : tensor<10x4xf32>) {
  ^bb0(%in0: bf16, %in1: bf16, %out: f32):
    %a = arith.extf %in0 : bf16 to f32
    %b = arith.extf %in1 : bf16 to f32
    %c = arith.mulf %a, %b : f32
    %d = arith.addf %out, %c : f32
    linalg.yield %d : f32
  } -> tensor<10x4xf32>
  return %result : tensor<10x4xf32>
}

// CHECK-LABEL: @swap_bf16_matmul
// CHECK-SAME:      %[[SRC:.*]]: tensor<4x8xbf16>, %[[FIL:.*]]: tensor<8x4xbf16>
// Small matmul on source directly:
// CHECK:       linalg.generic
// CHECK-SAME:      ins(%[[SRC]], %[[FIL]] : tensor<4x8xbf16>, tensor<8x4xbf16>)
// CHECK:       -> tensor<4x4xf32>
// Output scatter with f32 result:
// CHECK:       tensor.insert_slice
// CHECK-SAME:      [1, 0] [4, 4] [2, 1]
// CHECK-SAME:      tensor<4x4xf32> into tensor<10x4xf32>

// -----

// No transformation: insert_slice has multiple users.
#map_id = affine_map<(d0, d1) -> (d0, d1)>
#map_mm_a3 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map_mm_b3 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map_mm_c3 = affine_map<(d0, d1, d2) -> (d0, d1)>
func.func @no_swap_multiple_users(
    %src: tensor<4x8xf16>,
    %filter: tensor<8x4xf16>) -> (tensor<10x4xf32>, tensor<10x8xf16>) {
  %cst = arith.constant dense<0.000000e+00> : tensor<10x8xf16>
  %scattered = tensor.insert_slice %src into %cst[1, 0] [4, 8] [2, 1]
    : tensor<4x8xf16> into tensor<10x8xf16>
  %cst_f32 = arith.constant 0.000000e+00 : f32
  %empty = tensor.empty() : tensor<10x4xf32>
  %fill = linalg.fill ins(%cst_f32 : f32) outs(%empty : tensor<10x4xf32>) -> tensor<10x4xf32>
  %result = linalg.generic {
      indexing_maps = [#map_mm_a3, #map_mm_b3, #map_mm_c3],
      iterator_types = ["parallel", "parallel", "reduction"]}
    ins(%scattered, %filter : tensor<10x8xf16>, tensor<8x4xf16>)
    outs(%fill : tensor<10x4xf32>) {
  ^bb0(%in0: f16, %in1: f16, %out: f32):
    %a = arith.extf %in0 : f16 to f32
    %b = arith.extf %in1 : f16 to f32
    %c = arith.mulf %a, %b : f32
    %d = arith.addf %out, %c : f32
    linalg.yield %d : f32
  } -> tensor<10x4xf32>
  return %result, %scattered : tensor<10x4xf32>, tensor<10x8xf16>
}

// CHECK-LABEL: @no_swap_multiple_users
// CHECK:       tensor.insert_slice
// CHECK:       linalg.generic
// CHECK-NOT:   tensor.insert_slice
