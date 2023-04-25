// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/TransformDialectStrategies/GPU/ConvolutionPaddedImplicitGemmStrategy.h"

#include "iree-dialects/Dialect/LinalgTransform/StructuredTransformOpsExt.h"
#include "iree/compiler/Codegen/Common/TransformExtensions/CommonExtensions.h"
#include "iree/compiler/Codegen/LLVMGPU/TransformExtensions/LLVMGPUExtensions.h"
#include "iree/compiler/Codegen/TransformDialectStrategies/Common/Common.h"
#include "iree/compiler/Codegen/TransformDialectStrategies/GPU/Common.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"

using namespace mlir;

#define DEBUG_TYPE "iree-transform-builder"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")

// TODO: significantly better namespacing.
using iree_compiler::IREE::transform_dialect::ApplyPatternsOp;
using iree_compiler::IREE::transform_dialect::ApplyPatternsOpPatterns;
using iree_compiler::IREE::transform_dialect::ApplyPatternsToNestedOp;
using iree_compiler::IREE::transform_dialect::ForallToWorkgroupOp;
using iree_compiler::IREE::transform_dialect::GpuDistributeSharedMemoryCopyOp;
using iree_compiler::IREE::transform_dialect::HoistStaticAllocOp;
using iree_compiler::IREE::transform_dialect::IREEBufferizeOp;
using iree_compiler::IREE::transform_dialect::IREEEliminateEmptyTensorsOp;
using iree_compiler::IREE::transform_dialect::
    IREEEraseHALDescriptorTypeFromMemRefOp;
using iree_compiler::IREE::transform_dialect::
    IREEPopulateWorkgroupCountRegionUsingNumThreadsSliceOp;
using iree_compiler::IREE::transform_dialect::MapNestedForallToGpuThreadsOp;
using iree_compiler::IREE::transform_dialect::PromoteOperandsOp;
using iree_compiler::IREE::transform_dialect::ShareForallOperandsOp;
using iree_compiler::IREE::transform_dialect::VectorToMMAConversionOp;
using iree_compiler::IREE::transform_dialect::VectorToWarpExecuteOnLane0Op;
using iree_compiler::IREE::transform_dialect::VectorWarpDistributionOp;
using transform::ConvertConv2DToImg2ColOp;
using transform::FuseIntoContainingOp;
using transform::MatchOp;
using transform::PrintOp;
using transform::RewriteInDestinationPassingStyleOp;
using transform::ScalarizeOp;
using transform::SequenceOp;
using transform::SplitHandlesOp;
using transform::VectorizeOp;
using transform_ext::StructuredOpMatcher;

using iree_compiler::buildTileReductionUsingScfForeach;
using iree_compiler::gpu::AbstractReductionStrategy;
using iree_compiler::gpu::adjustNumberOfWarpsForBlockShuffle;
using iree_compiler::gpu::build1DSplittingStrategyWithOptionalThreadMapping;
using iree_compiler::gpu::buildCommonTrailingStrategy;
using iree_compiler::gpu::buildDistributeVectors;
using iree_compiler::gpu::ConvolutionPaddedImplicitGemmStrategy;
using iree_compiler::gpu::kCudaMaxNumThreads;
using iree_compiler::gpu::kCudaMaxVectorLoadBitWidth;
using iree_compiler::gpu::kCudaWarpSize;
using iree_compiler::gpu::ReductionConfig;
using iree_compiler::gpu::scaleUpByBitWidth;

ConvolutionPaddedImplicitGemmStrategy
mlir::iree_compiler::gpu::ConvolutionPaddedImplicitGemmStrategy::create(
    MLIRContext *context,
    const transform_ext::MatchedConvolutionCaptures &captures,
    const ConvolutionConfig &convolutionConfig) {
  ConvolutionPaddedImplicitGemmStrategy strategy(context, captures);
  strategy.configure(convolutionConfig);
  LLVM_DEBUG(DBGS() << "use GPU convolution implicit gemm strategy\n");
  return strategy;
}

void mlir::iree_compiler::gpu::ConvolutionPaddedImplicitGemmStrategy::configure(
    const ConvolutionConfig &convolutionConfig) {
  isSpirv = convolutionConfig.isSpirv;
  int64_t maxNumThreadsToUse = convolutionConfig.maxNumThreads;
  assert(maxNumThreadsToUse > 0 && "maxNumThreadsToUse must be > 0");
  assert(maxNumThreadsToUse >= convolutionConfig.subgroupSize &&
         "need at least a warp?");

  // llvm::errs() << "\n";
  // llvm::errs() << "\n";
  // llvm::interleaveComma(captures.convolutionDims.batch, llvm::errs() <<
  // "Batch: "); llvm::errs() << "\n";
  // llvm::interleaveComma(captures.convolutionDims.outputImage, llvm::errs() <<
  // "OutputImage: "); llvm::errs() << "\n";
  // llvm::interleaveComma(captures.convolutionDims.outputChannel, llvm::errs()
  // << "OutputChannel: "); llvm::errs() << "\n";
  // llvm::interleaveComma(captures.convolutionDims.filterLoop, llvm::errs() <<
  // "FilterLoop: "); llvm::errs() << "\n";
  // llvm::interleaveComma(captures.convolutionDims.inputChannel, llvm::errs()
  // << "InputChannel: "); llvm::errs() << "\n";
  // llvm::interleaveComma(captures.convolutionDims.depth, llvm::errs() <<
  // "Depth: "); llvm::errs() << "\n";
  // llvm::interleaveComma(captures.convolutionDims.strides, llvm::errs() <<
  // "Strides: "); llvm::errs() << "\n";
  // llvm::interleaveComma(captures.convolutionDims.dilations, llvm::errs() <<
  // "Dilations: "); llvm::errs() << "\n"; llvm::errs() << "\n";

  if (captures.convolutionDims.inputChannel.size() == 1 &&
      captures.convolutionDims.outputChannel.size() == 1) {
    // Block-level
    // ===========

    // Batch dimension
    for (int i = 0, e = captures.convolutionDims.batch.size(); i < e; i++)
      workgroupTileSizes.push_back(1);

    // Extra Outer Image dimensions
    for (int i = 0, e = captures.convolutionDims.outputImage.size() - 1; i < e;
         i++)
      workgroupTileSizes.push_back(1);

    isNchw = captures.convolutionDims.outputChannel[0] <
             captures.convolutionDims.outputImage[0];

    int channelSize = 1;
    for (auto dim : captures.convolutionDims.outputChannel)
      channelSize *= captures.convolutionOpSizes[dim];
    int imageSize =
        captures
            .convolutionOpSizes[captures.convolutionDims.outputImage.back()];

    int mSize, nSize;
    if (isNchw) {
      mSize = channelSize;
      nSize = imageSize;
    } else {
      mSize = imageSize;
      nSize = channelSize;
    }

    int kSize = 1;
    for (auto dim : captures.convolutionDims.inputChannel)
      kSize *= captures.convolutionOpSizes[dim];

    LLVM_DEBUG(DBGS() << "M size:" << mSize << ", " << mSize % 32 << "\n");
    LLVM_DEBUG(DBGS() << "N size:" << nSize << ", " << nSize % 32 << "\n");
    LLVM_DEBUG(DBGS() << "K size:" << kSize << ", " << kSize % 32 << "\n");

    int64_t mTileSize = 32;
    int64_t nTileSize = 64;
    if (isNchw) {
      mTileSize = 64;
      nTileSize = 32;
    }

    while (mSize % mTileSize != 0) mTileSize /= 2;
    workgroupTileSizes.push_back(mTileSize);

    while (nSize % nTileSize != 0) nTileSize /= 2;
    workgroupTileSizes.push_back(nTileSize);

    tileM = mTileSize > nTileSize;
    int64_t threadTile = tileM ? mTileSize : nTileSize;

    int64_t im2colTile = isNchw ? nTileSize : mTileSize;

    // Thread-level
    // ============
    numThreadsXInBlock =
        std::min(maxNumThreadsToUse,
                 // 2 * convolutionConfig.subgroupSize);
                 iree_compiler::nextMultipleOf(threadTile / 2,
                                               convolutionConfig.subgroupSize));
    numThreadsXToDistribute = std::min(threadTile, numThreadsXInBlock);
    numThreadsXForIm2Col = std::min(im2colTile, numThreadsXInBlock);
    numWarpsXInBlock = numThreadsXInBlock / convolutionConfig.subgroupSize;

    // Reduction tile size
    innerLoopTileSize = kSize % 32 != 0 ? 16 : 32;

    // Build tile size vectors.

    reductionLoopTileSizes =
        SmallVector<int64_t>(captures.convolutionDims.batch.size(), 0);
    if (isNchw) {
      reductionLoopTileSizes.append(
          captures.convolutionDims.outputChannel.size(), 0);
      reductionLoopTileSizes.push_back(innerLoopTileSize);
      reductionLoopTileSizes.append(
          captures.convolutionDims.filterLoop.size() - 1, 0);
      reductionLoopTileSizes.append(captures.convolutionDims.outputImage.size(),
                                    0);
      reductionLoopTileSizes.push_back(1);
    } else {
      reductionLoopTileSizes.append(captures.convolutionDims.outputImage.size(),
                                    0);
      reductionLoopTileSizes.append(captures.convolutionDims.filterLoop.size(),
                                    1);
      reductionLoopTileSizes.append(
          captures.convolutionDims.outputChannel.size(), 0);
      reductionLoopTileSizes.push_back(innerLoopTileSize);
    }

    im2ColThreadTileSizes =
        SmallVector<int64_t>(captures.convolutionDims.batch.size(), 0);
    if (isNchw) {
      im2ColThreadTileSizes.append(captures.convolutionDims.inputChannel.size(),
                                   0);
      im2ColThreadTileSizes.append(
          captures.convolutionDims.filterLoop.size() - 1, 0);
    }
    im2ColThreadTileSizes.append(
        captures.convolutionDims.outputImage.size() - 1, 0);
    im2ColThreadTileSizes.push_back(numThreadsXForIm2Col);

    inputPadThreadTileSizes =
        SmallVector<int64_t>(captures.convolutionDims.batch.size(), 0);
    if (isNchw) {
      inputPadThreadTileSizes.append(
          captures.convolutionDims.inputChannel.size(), 0);
      inputPadThreadTileSizes.append(
          captures.convolutionDims.filterLoop.size() - 1, 0);
    }
    inputPadThreadTileSizes.append(
        captures.convolutionDims.outputImage.size() - 1, 0);
    inputPadThreadTileSizes.push_back(numThreadsXForIm2Col);

    elementwiseThreadTileSizes =
        SmallVector<int64_t>(captures.convolutionDims.batch.size(), 0);
    if (isNchw) elementwiseThreadTileSizes.push_back(0);
    if (!tileM) elementwiseThreadTileSizes.push_back(0);
    elementwiseThreadTileSizes.append(
        captures.convolutionDims.outputImage.size() - 1, 0);
    elementwiseThreadTileSizes.push_back(numThreadsXToDistribute);

    matmulWarpTileSizes =
        SmallVector<int64_t>(captures.convolutionDims.batch.size(), 0);
    if (isNchw) matmulWarpTileSizes.push_back(0);
    if (!tileM) matmulWarpTileSizes.push_back(0);
    matmulWarpTileSizes.append(captures.convolutionDims.outputImage.size() - 1,
                               0);
    matmulWarpTileSizes.push_back(numWarpsXInBlock);
  } else if (!captures.convolutionDims.filterLoop.empty() &&
             (captures.convolutionDims.inputChannel.size() == 2 ||
              captures.convolutionDims.outputChannel.size() == 2)) {
    // Block-level
    // ===========

    // Batch dimension
    for (int i = 0, e = captures.convolutionDims.batch.size(); i < e; i++)
      workgroupTileSizes.push_back(1);

    // Extra Outer Channel dimension
    for (int i = 0, e = captures.convolutionDims.outputChannel.size() - 1;
         i < e; i++)
      workgroupTileSizes.push_back(1);

    // Extra Outer Image dimensions
    for (int i = 0, e = captures.convolutionDims.outputImage.size() - 1; i < e;
         i++)
      workgroupTileSizes.push_back(1);

    int channelSize = 1;
    int imageSize =
        captures
            .convolutionOpSizes[captures.convolutionDims.outputImage.back()];
    for (auto dim : captures.convolutionDims.outputChannel)
      channelSize *= captures.convolutionOpSizes[dim];

    int mSize = imageSize;
    int nSize = channelSize;

    int kSize =
        captures
            .convolutionOpSizes[captures.convolutionDims.inputChannel.back()];

    LLVM_DEBUG(DBGS() << "M size:" << mSize << ", " << mSize % 32 << "\n");
    LLVM_DEBUG(DBGS() << "N size:" << nSize << ", " << nSize % 32 << "\n");
    LLVM_DEBUG(DBGS() << "K size:" << kSize << ", " << kSize % 32 << "\n");

    int64_t mTileSize = 128;
    // int64_t nTileSize = nSize;

    while (mSize % mTileSize != 0) mTileSize /= 2;
    workgroupTileSizes.push_back(mTileSize);

    // workgroupTileSizes.push_back(nTileSize);

    int64_t threadTile = mTileSize;
    int64_t im2colTile = mTileSize;

    // Thread-level
    // ============
    numThreadsXInBlock =
        std::min(maxNumThreadsToUse,
                 // 2 * convolutionConfig.subgroupSize);
                 iree_compiler::nextMultipleOf(threadTile / 2,
                                               convolutionConfig.subgroupSize));
    numThreadsXToDistribute = std::min(threadTile, numThreadsXInBlock);
    numThreadsXForIm2Col = std::min(im2colTile, numThreadsXInBlock);
    numWarpsXInBlock = numThreadsXInBlock / convolutionConfig.subgroupSize;

    // Reduction tile size
    innerLoopTileSize = kSize % 32 != 0 ? 16 : 32;

    // Build tile size vectors.

    if (captures.convolutionDims.inputChannel.size() > 1) {
      reductionLoopTileSizes = SmallVector<int64_t>(
          captures.convolutionDims.batch.size() +
              captures.convolutionDims.outputChannel.size() - 1,
          0);
      reductionLoopTileSizes.push_back(1);
      reductionLoopTileSizes.append(captures.convolutionDims.filterLoop.size(),
                                    1);
    } else {
      reductionLoopTileSizes = SmallVector<int64_t>(
          captures.convolutionDims.batch.size() +
              captures.convolutionDims.outputImage.size() +
              captures.convolutionDims.outputChannel.size() - 1,
          0);
      reductionLoopTileSizes.append(captures.convolutionDims.filterLoop.size(),
                                    1);
      reductionLoopTileSizes.push_back(0);
      reductionLoopTileSizes.push_back(innerLoopTileSize);
    }

    if (captures.convolutionDims.inputChannel.size() > 1) {
      im2ColThreadTileSizes = SmallVector<int64_t>(
          captures.convolutionDims.batch.size() +
              captures.convolutionDims.filterLoop.size() +
              captures.convolutionDims.outputImage.size() - 1 +
              captures.convolutionDims.inputChannel.size() - 1,
          0);
      im2ColThreadTileSizes.push_back(numThreadsXForIm2Col);
    } else {
      im2ColThreadTileSizes = SmallVector<int64_t>(
          captures.convolutionDims.batch.size() +
              captures.convolutionDims.outputImage.size() - 1 +
              captures.convolutionDims.inputChannel.size() - 1,
          0);
      im2ColThreadTileSizes.push_back(numThreadsXForIm2Col);
    }

    inputPadThreadTileSizes = SmallVector<int64_t>(
        captures.convolutionDims.batch.size() +
            captures.convolutionDims.outputImage.size() - 1 +
            captures.convolutionDims.inputChannel.size() - 1,
        0);
    inputPadThreadTileSizes.push_back(numThreadsXForIm2Col);

    elementwiseThreadTileSizes = SmallVector<int64_t>(
        captures.convolutionDims.batch.size() +
            captures.convolutionDims.outputImage.size() - 1 +
            captures.convolutionDims.outputChannel.size() - 1,
        0);
    elementwiseThreadTileSizes.push_back(numThreadsXToDistribute);

    matmulWarpTileSizes = SmallVector<int64_t>(
        captures.convolutionDims.batch.size() +
            captures.convolutionDims.outputImage.size() - 1 +
            captures.convolutionDims.outputChannel.size() - 1,
        0);
    if (captures.convolutionDims.inputChannel.size() > 1) {
      matmulWarpTileSizes.append(captures.convolutionDims.filterLoop.size(), 0);
      matmulWarpTileSizes.append(
          captures.convolutionDims.inputChannel.size() - 1, 0);
    }
    matmulWarpTileSizes.push_back(numWarpsXInBlock);
  } else {
    assert(false && "should not have matched padded implicit gemm yet");
  }
}

/// Builds the transform IR tiling reductions for CUDA targets. Supports
/// reductions in the last dimension, with optional leading and trailing
/// elementwise operations.
void mlir::iree_compiler::gpu::buildConvolutionPaddedImplicitGemmStrategy(
    ImplicitLocOpBuilder &b, Value variantH,
    const ConvolutionPaddedImplicitGemmStrategy &strategy) {
  // LLVM_DEBUG(b.create<PrintOp>(variantH));

  ApplyPatternsOpPatterns emptyConfiguration;
  auto pdlOperationType = pdl::OperationType::get(b.getContext());

  // Step 1. Call the matcher. Note that this is the same matcher as used to
  // trigger this compilation path, so it must always apply.
  b.create<transform_ext::RegisterMatchCallbacksOp>();
  auto [maybePadH, maybeFillH, convolutionH, maybeTrailingH] =
      unpackRegisteredMatchCallback<4>(
          b, "convolution", transform::FailurePropagationMode::Propagate,
          variantH);

  // Step 2. Create the block/mapping tiling level and fuse.
  auto [fusionTargetH, fusionGroupH] =
      buildSelectFirstNonEmpty(b, maybeTrailingH, convolutionH);
  ArrayRef<Attribute> allBlocksRef(strategy.allBlockAttrs);
  TileToForallAndFuseAndDistributeResult tileResult =
      buildTileFuseDistToForallWithTileSizes(
          /*builder=*/b,
          /*isolatedParentOpH=*/variantH,
          /*rootH=*/fusionTargetH,
          /*opsToFuseH=*/fusionGroupH,
          /*tileSizes=*/
          getAsOpFoldResult(b.getI64ArrayAttr(strategy.workgroupTileSizes)),
          /*threadDimMapping=*/
          b.getArrayAttr(
              allBlocksRef.take_front(strategy.workgroupTileSizes.size())));

  // Handle the workgroup count region.
  b.create<IREEPopulateWorkgroupCountRegionUsingNumThreadsSliceOp>(
      tileResult.forallH);

  /// The previous fill handle gets invalidated so we match it again.
  Value newFillH =
      b.create<MatchOp>(variantH, linalg::FillOp::getOperationName());
  maybeFillH =
      b.create<FuseIntoContainingOp>(newFillH, tileResult.forallH).getResult();
  maybePadH =
      b.create<FuseIntoContainingOp>(maybePadH, tileResult.forallH).getResult();

  // LLVM_DEBUG(b.create<PrintOp>(variantH));

  /// Perform a pass of canonicalization + enabling after fusion.
  variantH = buildCanonicalizationAndEnablingTransforms(b, emptyConfiguration,
                                                        variantH);

  // Step 3. Apply im2col patterns.
  auto [blockConvH, maybeBlockTrailingH] = buildSelectFirstNonEmpty(
      b, tileResult.resultingFusedOpsHandles.front(), tileResult.tiledOpH);
  auto img2colWorkgroupOp = b.create<ConvertConv2DToImg2ColOp>(
      blockConvH,
      /*noCollapseFilter=*/true, /*noCollapseOutput=*/true);
  Value img2colH = img2colWorkgroupOp.getImg2colTensor();
  Value matmulH = img2colWorkgroupOp.getTransformed();

  // LLVM_DEBUG(b.create<PrintOp>(variantH));

  // Step 4. Bubble reshapes introduced by im2col to the boundaries of the
  // kernel.
  Value funcH = b.create<MatchOp>(variantH, func::FuncOp::getOperationName());
  // ApplyPatternsOpPatterns expandConfig;
  // configuration.bubbleExpand = true;
  // b.create<ApplyPatternsOp>(funcH, expandConfig);
  // ApplyPatternsOpPatterns collapseConfig;
  // configuration.bubbleCollapse = true;
  // b.create<ApplyPatternsOp>(funcH, collapseConfig);
  variantH = buildCanonicalizationAndEnablingTransforms(b, emptyConfiguration,
                                                        variantH);

  // Step 5. Tile the reduction loop
  SmallVector<Type> resultTypes(strategy.getNumReductionLoops() + 1,
                                pdlOperationType);
  auto tileToScfForOp = b.create<transform::TileToScfForOp>(
      TypeRange(resultTypes), matmulH, ValueRange{},
      strategy.getReductionLoopTileSizes());
  auto matmulLoopK = tileToScfForOp.getTiledLinalgOp();

  Value tiledImg2colH = img2colH;
  for (auto loopIndex = 0; loopIndex < strategy.getNumReductionLoops();
       loopIndex++) {
    tiledImg2colH = b.create<FuseIntoContainingOp>(
                         tiledImg2colH, tileToScfForOp.getLoops()[loopIndex])
                        .getResult();
  }
  variantH = buildCanonicalizationAndEnablingTransforms(b, emptyConfiguration,
                                                        variantH);

  // LLVM_DEBUG(b.create<PrintOp>(variantH));

  // Step 6. Promote to shared memory
  auto promoteOperandsOp = b.create<PromoteOperandsOp>(
      TypeRange{pdlOperationType, pdlOperationType}, matmulLoopK,
      b.getDenseI64ArrayAttr(
          ArrayRef<int64_t>{strategy.getImplicitGemmFilterOperandIndex()}));
  Value promotedMatmulH = promoteOperandsOp.getResult()[0];

  // LLVM_DEBUG(b.create<PrintOp>(variantH));

  // Step 7. Tile img2col, fill, and trailing elementwise to threads
  iree_compiler::buildTileFuseDistToForallWithNumThreads(
      /*b=*/b,
      /*isolatedParentOpH=*/variantH,
      /*rootH=*/tiledImg2colH,
      /*opsHToFuse=*/{},
      /*numThreads=*/
      getAsOpFoldResult(b.getI64ArrayAttr(strategy.getIm2ColThreadTileSizes())),
      /*threadDimMapping=*/b.getArrayAttr({strategy.allThreadAttrs.front()}));

  iree_compiler::buildTileFuseDistToForallWithNumThreads(
      /*b=*/b,
      /*isolatedParentOpH=*/variantH,
      /*rootH=*/maybeBlockTrailingH,
      /*opsHToFuse=*/{},
      /*numThreads=*/
      getAsOpFoldResult(
          b.getI64ArrayAttr(strategy.getElementwiseThreadTileSizes())),
      /*threadDimMapping=*/b.getArrayAttr({strategy.allThreadAttrs.front()}));

  // Match the fill again I guess...
  newFillH = b.create<MatchOp>(variantH, linalg::FillOp::getOperationName());
  iree_compiler::buildTileFuseDistToForallWithNumThreads(
      /*b=*/b,
      /*isolatedParentOpH=*/variantH,
      /*rootH=*/newFillH,
      /*opsHToFuse=*/{},
      /*numThreads=*/
      getAsOpFoldResult(
          b.getI64ArrayAttr(strategy.getElementwiseThreadTileSizes())),
      /*threadDimMapping=*/b.getArrayAttr({strategy.allThreadAttrs.front()}));

  // LLVM_DEBUG(b.create<PrintOp>(variantH));

  // Step 9. Tile matmul to warps
  iree_compiler::buildTileFuseDistToForallWithNumThreads(
      /*b=*/b,
      /*isolatedParentOpH=*/variantH,
      /*rootH=*/promotedMatmulH,
      /*opsHToFuse=*/{},
      /*numWarps=*/
      getAsOpFoldResult(b.getI64ArrayAttr(strategy.getMatmulWarpTileSizes())),
      /*threadDimMapping=*/b.getArrayAttr({strategy.allWarpAttrs.front()}));

  // LLVM_DEBUG(b.create<PrintOp>(variantH));

  // Step 8. Tile and rewrite tensor pad
  auto newPadH = b.create<MatchOp>(variantH, tensor::PadOp::getOperationName());
  auto padTilingRes = iree_compiler::buildTileFuseDistToForallWithNumThreads(
      /*b=*/b,
      /*isolatedParentOpH=*/variantH,
      /*rootH=*/newPadH,
      /*opsHToFuse=*/{},
      /*numThreads=*/
      getAsOpFoldResult(
          b.getI64ArrayAttr(strategy.getInputPadThreadTileSizes())),
      /*threadDimMapping=*/b.getArrayAttr({strategy.allThreadAttrs.front()}));
  b.create<RewriteInDestinationPassingStyleOp>(TypeRange{pdlOperationType},
                                               padTilingRes.tiledOpH);
  b.create<IREEEliminateEmptyTensorsOp>(variantH);

  // LLVM_DEBUG(b.create<PrintOp>(variantH));

  // Step 8. Vectorize and unroll to wmma sizes
  funcH = b.create<MatchOp>(variantH, func::FuncOp::getOperationName());

  ApplyPatternsOpPatterns vectorizeConfiguration;
  vectorizeConfiguration.rankReducingLinalg = true;
  vectorizeConfiguration.rankReducingVector = true;
  b.create<ApplyPatternsOp>(funcH, vectorizeConfiguration);
  funcH = b.create<VectorizeOp>(funcH, /*vectorizePadding=*/false,
                                /*vectorizeExtract=*/true);
  // LLVM_DEBUG(b.create<PrintOp>(variantH));

  // Basically a hack to find the parent forall loop of the matmul for wmma
  // unrolling.
  auto forallOpsH =
      b.create<MatchOp>(variantH, scf::ForallOp::getOperationName());
  int numLoops = 4;
  int resultPos = 2;
  if (strategy.captures.maybeFillElementalTypeBitWidth > 0) {
    numLoops++;
    resultPos++;
  }
  if (strategy.captures.maybeTrailingOutputElementalTypeBitWidth > 0)
    numLoops++;
  Value matmulLoop =
      b.create<SplitHandlesOp>(forallOpsH, numLoops)->getResult(resultPos);

  ApplyPatternsOpPatterns unrollConfiguration;
  if (strategy.getIsSpirv())
    unrollConfiguration.unrollVectorsGpuCoopMat = true;
  else
    unrollConfiguration.unrollVectorsGpuWmma = true;
  // LLVM_DEBUG(b.create<PrintOp>(matmulLoop));
  b.create<ApplyPatternsToNestedOp>(matmulLoop, unrollConfiguration);

  // LLVM_DEBUG(b.create<PrintOp>(variantH));

  // Step 9. Bufferize
  ApplyPatternsOpPatterns foldConfiguration;
  foldConfiguration.foldReassociativeReshapes = true;
  b.create<ApplyPatternsOp>(funcH, foldConfiguration);

  b.create<IREEEliminateEmptyTensorsOp>(variantH);
  funcH = b.create<MatchOp>(variantH, func::FuncOp::getOperationName());

  ApplyPatternsOpPatterns eraseConfiguration;
  eraseConfiguration.eraseUnnecessaryTensorOperands = true;
  b.create<ApplyPatternsOp>(funcH, eraseConfiguration);

  // LLVM_DEBUG(b.create<PrintOp>(variantH));

  variantH = b.create<IREEBufferizeOp>(variantH, /*targetGPU=*/true);
  variantH = buildCanonicalizationAndEnablingTransforms(b, emptyConfiguration,
                                                        variantH);

  // LLVM_DEBUG(b.create<PrintOp>(variantH));

  // Step 12. Post-bufferization mapping to blocks and threads
  funcH = b.create<MatchOp>(variantH, func::FuncOp::getOperationName());
  funcH = buildMapToBlockAndThreads(b, funcH, strategy.getNumThreadsInBlock(),
                                    strategy.getNumWarpsInBlock());
  b.create<HoistStaticAllocOp>(funcH);
  // LLVM_DEBUG(b.create<PrintOp>(variantH));
  funcH = b.create<GpuDistributeSharedMemoryCopyOp>(TypeRange{pdlOperationType},
                                                    funcH);
  ApplyPatternsOpPatterns distributeConfiguration;
  eraseConfiguration.foldMemrefAliases = true;
  variantH = buildCanonicalizationAndEnablingTransforms(
      b, distributeConfiguration, variantH);
  if (!strategy.getIsSpirv())
    b.create<IREEEraseHALDescriptorTypeFromMemRefOp>(funcH);
  b.create<VectorToMMAConversionOp>(funcH, /*useMmaSync=*/false,
                                    /*useWmma=*/true);
  variantH = buildCanonicalizationAndEnablingTransforms(b, emptyConfiguration,
                                                        variantH);

  // LLVM_DEBUG(b.create<PrintOp>(variantH));
}
