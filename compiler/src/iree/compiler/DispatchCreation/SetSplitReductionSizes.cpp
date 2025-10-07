// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/TargetUtils/ConfigUtils.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtInterfaces.h"
#include "iree/compiler/Dialect/LinalgExt/Utils/Utils.h"
#include "iree/compiler/DispatchCreation/Passes.h"
#include "llvm/Support/DebugLog.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"

#define DEBUG_TYPE "iree-dispatch-creation-set-split-reduction-sizes"

namespace mlir::iree_compiler::DispatchCreation {

#define GEN_PASS_DEF_SETSPLITREDUCTIONSIZESPASS
#include "iree/compiler/DispatchCreation/Passes.h.inc"

static SmallVector<int64_t> getStaticReductionDimSizes(linalg::LinalgOp op) {
  SmallVector<int64_t> dimSizes;
  for (auto [loopRange, loopType] :
       llvm::zip_equal(op.getStaticLoopRanges(), op.getIteratorTypesArray())) {
    if (loopType == utils::IteratorType::reduction) {
      dimSizes.push_back(loopRange);
    }
  }
  return dimSizes;
}

static std::optional<SmallVector<int64_t>> getReductionDimSizes(Operation *Op) {
  auto fusionOp = dyn_cast<IREE::LinalgExt::LinalgFusionOpInterface>(Op);
  if (!fusionOp) {
    LDBG() << "skipping op; not a LinalgFusionOpInterface op";
    return std::nullopt;
  }
  SmallVector<int64_t> loopRanges = fusionOp.getStaticLoopRanges();

  auto tilingInterfaceOp = dyn_cast<TilingInterface>(Op);
  if (!tilingInterfaceOp) {
    LDBG() << "skipping op; not a TilingInterface op";
    return std::nullopt;
  }

  SmallVector<utils::IteratorType> iters =
      tilingInterfaceOp.getLoopIteratorTypes();
  SmallVector<int64_t> reductionDimSizes;
  for (auto [range, it] : llvm::zip_equal(loopRanges, iters)) {
    if (it == utils::IteratorType::reduction) {
      reductionDimSizes.push_back(range);
    }
  }
  return reductionDimSizes;
}

static std::optional<int64_t>
findSmallestFactorWithLowerBound(int64_t x, int64_t lowerBound) {
  assert(x > 0);
  assert(lowerBound > 0);
  // We expect all numbers here to be relatively small, so just do trial
  // division (with a limit just to be safe).
  static constexpr int64_t kMaxIterations = 1 << 15;
  int64_t upperBound = std::min(x, kMaxIterations);
  for (int64_t i = lowerBound; i <= upperBound; i++) {
    if (x % i == 0) {
      return i;
    }
  }
  return std::nullopt;
};

static FailureOr<SmallVector<int64_t>> getWorkgroupTileSizesForMatmulOrIGEMM(
    SmallVector<int64_t> bounds, ArrayRef<AffineMap> maps,
    ArrayRef<Value> operands, IREE::GPU::TargetAttr target, bool isGemm) {
  if (target.getWgp().getMma().empty())
    return failure();

  // Infer contraction dims (M, N, K, Batch)
  SmallVector<unsigned, 2> contractionM, contractionN, contractionK,
      contractionB;
  auto dims = mlir::linalg::inferContractionDims(maps);
  if (failed(dims))
    return failure();
  contractionM = dims->m;
  contractionN = dims->n;
  contractionK = dims->k;
  contractionB = dims->batch;

  if (contractionM.empty() || contractionN.empty() || contractionK.empty())
    return failure();

  // Build problem shape for MMA schedule inference
  SmallVector<int64_t> mDims, nDims, kDims, batchDims;
  for (auto d : contractionM)
    mDims.push_back(d);
  for (auto d : contractionN)
    nDims.push_back(d);
  for (auto d : contractionK)
    kDims.push_back(d);
  for (auto d : contractionB)
    batchDims.push_back(d);

  Value lhs = operands[0];
  Value rhs = operands[1];
  Value init = operands[2];

  Type lhsElemType = getElementTypeOrSelf(lhs);
  Type rhsElemType = getElementTypeOrSelf(rhs);
  Type initElemType = getElementTypeOrSelf(init);

  bool transposedLhs =
      kDims.back() !=
      llvm::cast<AffineDimExpr>(maps[0].getResults().back()).getPosition();
  bool transposedRhs =
      nDims.back() !=
      llvm::cast<AffineDimExpr>(maps[1].getResults().back()).getPosition();

  GPUMatmulShapeType problem{
      llvm::map_to_vector(mDims, [&](int64_t d) { return bounds[d]; }),
      llvm::map_to_vector(nDims, [&](int64_t d) { return bounds[d]; }),
      llvm::map_to_vector(kDims, [&](int64_t d) { return bounds[d]; }),
      llvm::map_to_vector(batchDims, [&](int64_t d) { return bounds[d]; }),
      lhsElemType,
      rhsElemType,
      initElemType};

  auto schedule = getMmaScheduleFromProblemAndTarget(
      target, problem, transposedLhs, transposedRhs, isGemm);
  if (!schedule)
    return failure();

  // Compute workgroup tile sizes.
  SmallVector<int64_t> workgroupTileSizes(bounds.size(), 0);

  // Batch dimensions → tile = 1
  for (int64_t b : contractionB)
    workgroupTileSizes[b] = 1;

  // M and N dimensions
  for (auto [i, mDim] : llvm::enumerate(mDims)) {
    workgroupTileSizes[mDim] =
        schedule->mSubgroupCounts[i] * schedule->mTileSizes[i];
    if (i == mDims.size() - 1)
      workgroupTileSizes[mDim] *= schedule->mSize;
  }
  for (auto [i, nDim] : llvm::enumerate(nDims)) {
    workgroupTileSizes[nDim] =
        schedule->nSubgroupCounts[i] * schedule->nTileSizes[i];
    if (i == nDims.size() - 1)
      workgroupTileSizes[nDim] *= schedule->nSize;
  }

  // K dimensions (typically reduction — tile to 1)
  for (int64_t k : contractionK)
    workgroupTileSizes[k] = 1;

  return workgroupTileSizes;
}

namespace {
struct SetSplitReductionSizesPass final
    : public impl::SetSplitReductionSizesPassBase<SetSplitReductionSizesPass> {
  using Base::Base;
  void runOnOperation() override {
    // Skip pass if no target is set.
    if (splitReductionTargetSize <= 0) {
      return;
    }
    getOperation()->walk([&](PartialReductionOpInterface tilingOp) {
      // If the op already has its attribute set, don't change it.
      if (IREE::LinalgExt::getSplitReductionSizes(tilingOp).has_value()) {
        return;
      }
      // Skip ops that aren't reductions.
      unsigned numReduction = llvm::count_if(
          tilingOp.getLoopIteratorTypes(),
          [](utils::IteratorType iteratorType) {
            return iteratorType == utils::IteratorType::reduction;
          });
      if (numReduction == 0) {
        return;
      }

      // --- Case 1: Outer reduction ---
      if (auto tileSizes = getOuterReductionSizes(tilingOp)) {
        IREE::LinalgExt::setSplitReductionAttribute(tilingOp, *tileSizes);
        return;
      }

      // --- Case 2: Generic weight backward conv ---
      if (auto tileSizes = getWeightBackwardReductionSizes(tilingOp)) {
        IREE::LinalgExt::setSplitReductionAttribute(tilingOp, *tileSizes);
        return;
      }
    });
  }

private:
  /// Determine split reduction sizes for outer-reduction ops. This is
  /// targeting reductions such as those that appear in batch normalization,
  /// which reduce over outer dimensions of a tensor.
  std::optional<SmallVector<int64_t>>
  getOuterReductionSizes(PartialReductionOpInterface op) const {
    SmallVector<utils::IteratorType> iters = op.getLoopIteratorTypes();
    if (iters.empty() || iters.front() != utils::IteratorType::reduction) {
      LDBG() << "skipping op; not outer-reduction";
      return std::nullopt;
    }

    std::optional<SmallVector<int64_t>> maybeSizes =
        getReductionDimSizes(op.getOperation());
    if (!maybeSizes) {
      return std::nullopt;
    }
    SmallVector<int64_t> opReductionSizes = std::move(*maybeSizes);

    int64_t currentSplitReductionSize = 1;
    SmallVector<int64_t> tileSizes(opReductionSizes.size());
    // Tile dimensions until we reach or exceed the target. Tile sizes must
    // divide the dimension size evenly, and we start with inner dimensions as
    // we prefer tiling those.
    for (int64_t i = tileSizes.size() - 1; i >= 0; i--) {
      int64_t remainingSize =
          llvm::divideCeil(splitReductionTargetSize, currentSplitReductionSize);
      int64_t dimSize = opReductionSizes[i];
      if (dimSize == ShapedType::kDynamic) {
        LDBG() << "skipping op; has dynamic reduction dims";
        return std::nullopt;
      }
      int64_t tileSize =
          findSmallestFactorWithLowerBound(dimSize, remainingSize)
              .value_or(dimSize);
      tileSizes[i] = tileSize;
      currentSplitReductionSize *= tileSize;
    }
    return tileSizes;
  }

  std::optional<SmallVector<int64_t>>
  getWeightBackwardReductionSizes(PartialReductionOpInterface op) const {
    // First check if the input op is a convolution with CHWN layout.
    auto linalgOp = cast<linalg::LinalgOp>(op.getOperation());
    if (!linalgOp || !linalg::isaConvolutionOpInterface(linalgOp)) {
      LDBG() << "skipping op; not convolution";
      return std::nullopt;
    }

    FailureOr<mlir::linalg::ConvolutionDimensions> convDims =
        mlir::linalg::inferConvolutionDims(linalgOp);
    if (failed(convDims)) {
      LDBG() << "skipping op; failed to infer convolution dims";
      return std::nullopt;
    }

    OpOperand *input = linalgOp.getDpsInputOperand(0);
    OpOperand *filter = linalgOp.getDpsInputOperand(1);
    OpOperand *output = linalgOp.getDpsInitOperand(0);

    // AffineMap inputMap = linalgOp.getMatchingIndexingMap(input);
    // AffineMap filterMap = linalgOp.getMatchingIndexingMap(filter);
    AffineMap outputMap = linalgOp.getMatchingIndexingMap(output);

    Value inputVal = input->get();
    Value filterVal = filter->get();
    Value outputVal = output->get();

    ArrayRef<int64_t> inputShape =
        llvm::cast<ShapedType>(inputVal.getType()).getShape();
    ArrayRef<int64_t> filterShape =
        llvm::cast<ShapedType>(filterVal.getType()).getShape();
    ArrayRef<int64_t> outputShape =
        llvm::cast<ShapedType>(outputVal.getType()).getShape();

    // TODO(vivian): Support dynamic shapes.
    if (ShapedType::isDynamicShape(inputShape) ||
        ShapedType::isDynamicShape(filterShape) ||
        ShapedType::isDynamicShape(outputShape)) {
      LDBG() << "skipping op; has dynamic shape";
      return std::nullopt;
    }

    std::optional<int64_t> batchLastDim = outputMap.getResultPosition(
        getAffineDimExpr(convDims->batch.back(), outputMap.getContext()));
    if (!batchLastDim || batchLastDim.value() != outputShape.size() - 1) {
      LDBG() << "skipping op; not batch last layout";
      return std::nullopt;
    }

    std::optional<SmallVector<int64_t>> maybeSizes =
        getReductionDimSizes(op.getOperation());
    if (!maybeSizes) {
      LDBG() << "skipping op; failed to get reduction sizes";
      return std::nullopt;
    }
    SmallVector<int64_t> opReductionSizes = std::move(*maybeSizes);
    SmallVector<int64_t> tileSizes(opReductionSizes.size());

    // Try to prefetch the workgroup tile sizes.
    IREE::GPU::TargetAttr target = getGPUTargetAttr(op.getOperation());
    if (!target) {
      LDBG() << "skipping op; missing GPU target";
      return std::nullopt;
    }

    FailureOr<IREE::LinalgExt::IGEMMGenericConvDetails>
        igemmGenericConvDetails =
            IREE::LinalgExt::getIGEMMGenericConvDetails(linalgOp);
    if (failed(igemmGenericConvDetails)) {
      LDBG() << "skipping op; unsupported convolution type";
      return std::nullopt;
    }
    SmallVector<AffineMap> igemmContractionMaps =
        igemmGenericConvDetails->igemmContractionMaps;
    SmallVector<int64_t> igemmLoopBounds =
        igemmGenericConvDetails->igemmLoopBounds;
    SmallVector<Value> igemmOperands = igemmGenericConvDetails->igemmOperands;
    FailureOr<SmallVector<int64_t>> maybeWgTileSize =
        getWorkgroupTileSizesForMatmulOrIGEMM(
            igemmLoopBounds, igemmContractionMaps, igemmOperands, target,
            /*isGemm=*/false);
    if (failed(maybeWgTileSize)) {
      LDBG() << "skipping op; failed to prefetch workgroup tile sizes";
      return std::nullopt;
    }
    SmallVector<int64_t> wgTileSize = maybeWgTileSize.value();
    for (auto i : wgTileSize) {
      llvm::outs() << "HERE " << i << "\n";
    }

    return tileSizes;
  }
};
} // namespace
} // namespace mlir::iree_compiler::DispatchCreation
