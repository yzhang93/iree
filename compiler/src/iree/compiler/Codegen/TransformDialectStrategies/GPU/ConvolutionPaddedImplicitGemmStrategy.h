// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_TRANSFORM_DIALECT_STRATEGIES_GPU_CONVOLUTION_PADDED_IMPLICIT_GEMM_STRATEGY_H_
#define IREE_COMPILER_CODEGEN_TRANSFORM_DIALECT_STRATEGIES_GPU_CONVOLUTION_PADDED_IMPLICIT_GEMM_STRATEGY_H_

#include "iree/compiler/Codegen/TransformDialectStrategies/GPU/AbstractConvolutionStrategy.h"

namespace mlir {
namespace iree_compiler {
namespace gpu {

struct GPUModel;

class ConvolutionPaddedImplicitGemmStrategy
    : public AbstractConvolutionStrategy {
 public:
  static ConvolutionPaddedImplicitGemmStrategy create(
      MLIRContext *context,
      const transform_ext::MatchedConvolutionCaptures &captures,
      const ConvolutionConfig &convolutionConfig);

  ConvolutionPaddedImplicitGemmStrategy(
      const ConvolutionPaddedImplicitGemmStrategy &) = default;
  ConvolutionPaddedImplicitGemmStrategy &operator=(
      const ConvolutionPaddedImplicitGemmStrategy &) = default;

  SmallVector<int64_t> getNumThreadsInBlock() const override {
    return {numThreadsXInBlock, 1, 1};
  }

  SmallVector<int64_t> getNumWarpsInBlock() const override {
    return {numWarpsXInBlock, 1, 1};
  }

  int64_t getImplicitGemmFilterOperandIndex() const {
    if (captures.convolutionDims.outputChannel.back() <
        captures.convolutionDims.outputImage.back())
      return 0;
    return 1;
  }

  bool getIsSpirv() const { return isSpirv; }

  SmallVector<int64_t> getIm2ColThreadTileSizes() const {
    return im2ColThreadTileSizes;
  }

  SmallVector<int64_t> getInputPadThreadTileSizes() const {
    return inputPadThreadTileSizes;
  }

  SmallVector<int64_t> getElementwiseThreadTileSizes() const {
    return elementwiseThreadTileSizes;
  }

  SmallVector<int64_t> getMatmulWarpTileSizes() const {
    return matmulWarpTileSizes;
  }

  SmallVector<int64_t> getReductionLoopTileSizes() const {
    return reductionLoopTileSizes;
  }

  int64_t getNumReductionLoops() const {
    int numLoops = 0;
    for (auto t : reductionLoopTileSizes)
      if (t) numLoops++;
    return numLoops;
  }

 private:
  ConvolutionPaddedImplicitGemmStrategy(
      MLIRContext *context,
      const transform_ext::MatchedConvolutionCaptures &captures)
      : AbstractConvolutionStrategy(context, captures) {}

  void configure(const ConvolutionConfig &convolutionConfig);

  int64_t numThreadsXInBlock;
  int64_t numThreadsXToDistribute;
  int64_t numThreadsXForIm2Col;
  int64_t numWarpsXInBlock;
  int64_t innerLoopTileSize;

  SmallVector<int64_t> reductionLoopTileSizes;
  SmallVector<int64_t> im2ColThreadTileSizes;
  SmallVector<int64_t> inputPadThreadTileSizes;
  SmallVector<int64_t> elementwiseThreadTileSizes;
  SmallVector<int64_t> matmulWarpTileSizes;

  bool tileM = false;
  bool isNchw = false;
  bool isSpirv = false;
};

void buildConvolutionPaddedImplicitGemmStrategy(
    ImplicitLocOpBuilder &b, Value variantH,
    const ConvolutionPaddedImplicitGemmStrategy &strategy);

}  // namespace gpu
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_CODEGEN_TRANSFORM_DIALECT_STRATEGIES_GPU_CONVOLUTION_PADDED_IMPLICIT_GEMM_STRATEGY_H_
