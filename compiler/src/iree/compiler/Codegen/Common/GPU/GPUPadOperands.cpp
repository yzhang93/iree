// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/GPULoweringConfigUtils.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Transforms/Passes.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "mlir/IR/Attributes.h"
#include "llvm/Support/Casting.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_GPUPADOPERANDSPASS
#include "iree/compiler/Codegen/Common/GPU/Passes.h.inc"

namespace {

RankedTensorType getPaddedType(RankedTensorType inputType,
                               ArrayRef<OpFoldResult> lowPads,
                               ArrayRef<OpFoldResult> highPads) {
  SmallVector<int64_t> paddedShape;

  for (int i = 0; i < inputType.getRank(); ++i) {
    int64_t dim = inputType.getDimSize(i);

    // Default: dynamic
    int64_t newDim = ShapedType::kDynamic;

    // Try to extract pad values if they are constant
    auto getConstInt = [](OpFoldResult ofr) -> std::optional<int64_t> {
      if (auto attr = ofr.dyn_cast<Attribute>()) {
        if (auto intAttr = llvm::dyn_cast<mlir::IntegerAttr>(attr))
          return intAttr.getInt();
      }
      return std::nullopt;
    };

    auto lowPad = getConstInt(lowPads[i]);
    auto highPad = getConstInt(highPads[i]);

    if (dim != ShapedType::kDynamic && lowPad && highPad)
      newDim = dim + *lowPad + *highPad;

    paddedShape.push_back(newDim);
  }

  return RankedTensorType::get(paddedShape, inputType.getElementType());
}

struct SwapIm2ColAndPadPattern : OpRewritePattern<tensor::PadOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::PadOp padOp,
                                 PatternRewriter &rewriter) const override {
    // Match: tensor.pad where source is im2col op
    auto im2colOp = padOp.getSource().getDefiningOp<IREE::LinalgExt::Im2colOp>();
    if (!im2colOp)
      return failure();

    // Require ranked result on both padOp and input
    auto outputType = llvm::dyn_cast<RankedTensorType>(padOp.getResult().getType());
    if (!outputType || outputType.getRank() != 4)
      return failure();

    Value input = im2colOp.getOperand(0);
    auto inputType = llvm::dyn_cast<RankedTensorType>(input.getType());
    if (!inputType || inputType.getRank() != 4)
      return failure();

    int64_t inputRank = inputType.getRank();
    int64_t inputBatchDim = inputRank - 1; // batch is last dim in input
    int64_t outputBatchDim = 0;            // batch is first dim in output

    // Check if we are only padding the batch dimension of the output
    auto highPads = padOp.getMixedHighPad();
    auto lowPads = padOp.getMixedLowPad();
    if (highPads.size() != 4 || lowPads.size() != 4)
      return failure();

    // Only allow nonzero padding on output batch (dim 0)
    bool otherPadding =
        llvm::any_of(llvm::enumerate(highPads), [&](auto it) {
          return it.index() != outputBatchDim &&
                 !isZeroAttrOrValue(it.value());
        }) ||
        llvm::any_of(llvm::enumerate(lowPads), [&](auto it) {
          return it.index() != outputBatchDim &&
                 !isZeroAttrOrValue(it.value());
        });

    if (otherPadding)
      return failure();

    // Construct new padding on input's last dim (inputBatchDim)
    SmallVector<OpFoldResult> inputLowPads(inputRank,
                                           rewriter.getIndexAttr(0));
    SmallVector<OpFoldResult> inputHighPads(inputRank,
                                            rewriter.getIndexAttr(0));
    inputLowPads[inputBatchDim] = lowPads[outputBatchDim];
    inputHighPads[inputBatchDim] = highPads[outputBatchDim];

    // Create new pad op on input
    //auto paddedType = getPaddedType(inputType, inputLowPads, inputHighPads);
    // Force padded type to have static dim 16 in batch
    SmallVector<int64_t> newShape(inputType.getShape().begin(),
                                  inputType.getShape().end());
    newShape[inputBatchDim] = 16;
    auto paddedType = RankedTensorType::get(newShape, inputType.getElementType());

    auto paddedInput = rewriter.create<tensor::PadOp>(
        padOp.getLoc(), paddedType, input, inputLowPads, inputHighPads,
        padOp.getNofoldAttr() != nullptr);


    rewriter.inlineRegionBefore(padOp.getRegion(), paddedInput.getRegion(),
                                paddedInput.getRegion().begin());

    // Create new im2col with padded input
    auto elementType = outputType.getElementType();
    auto outputTensor = rewriter.create<tensor::EmptyOp>(im2colOp.getLoc(), outputType.getShape(), elementType);
    auto newIm2Col = rewriter.create<IREE::LinalgExt::Im2colOp>(
        im2colOp.getLoc(),
        paddedInput,
        outputTensor,
        im2colOp.getStrides(),
        im2colOp.getDilations(),
        im2colOp.getMixedKernelSize(),
        im2colOp.getMixedMOffset(),
        im2colOp.getMixedMStrides(),
        im2colOp.getMixedKOffset(),
        im2colOp.getMixedKStrides(),
        im2colOp.getBatchPos(),
        im2colOp.getMPos(),
        im2colOp.getKPos(),
        im2colOp.getInputKPerm()
    );

    // Replace tensor.dim ops that use the old im2colOp
    for (Operation *user : llvm::make_early_inc_range(im2colOp->getUsers())) {
      if (auto dimOp = dyn_cast<tensor::DimOp>(user)) {
        auto dimIndexAttr = dyn_cast_or_null<arith::ConstantOp>(
            dimOp.getIndex().getDefiningOp());
        if (!dimIndexAttr)
          continue;

        auto indexVal = dyn_cast<IntegerAttr>(dimIndexAttr.getValue());
        if (!indexVal || indexVal.getInt() != 0)
          continue;

        // Replace dim(im2colOp, 0) with dim(input, 3)
        rewriter.setInsertionPointAfter(dimOp);
        Value newIdx = rewriter.create<arith::ConstantIndexOp>(
            dimOp.getLoc(), inputBatchDim); // == 3
        Value newDim = rewriter.create<tensor::DimOp>(
            dimOp.getLoc(), input, newIdx);
        rewriter.replaceOp(dimOp, newDim);
      }
    }


    rewriter.replaceOp(padOp, newIm2Col.getResult(0));
    return success();
  }

private:
  // Helper to check if an OpFoldResult is constant 0
  static bool isZeroAttrOrValue(OpFoldResult ofr) {
    if (auto attr = ofr.dyn_cast<Attribute>()) {
      if (auto intAttr = llvm::dyn_cast<mlir::IntegerAttr>(attr))
        return intAttr.getValue().isZero();
    }
    return false;
  }
};

static LogicalResult padLinalgOpToStaticSizes(RewriterBase &rewriter,
                                              linalg::LinalgOp linalgOp,
                                              ArrayRef<int64_t> padding) {
  SmallVector<int64_t> paddingDims =
      llvm::to_vector(llvm::seq<int64_t>(0, linalgOp.getNumLoops()));
  SmallVector<bool> nofoldFlags(linalgOp.getNumDpsInputs(), /*nofold=*/false);
  SmallVector<Attribute> paddingValueAttributes;
  for (auto &operand : linalgOp->getOpOperands()) {
    Type elemType = getElementTypeOrSelf(operand.get().getType());
    paddingValueAttributes.push_back(rewriter.getZeroAttr(elemType));
  }

  auto options =
      linalg::LinalgPaddingOptions()
          .setPaddingDimensions(paddingDims)
          .setPaddingValues(paddingValueAttributes)
          .setPadToMultipleOf(padding)
          .setNofoldFlags(nofoldFlags)
          .setCopyBackOp(linalg::LinalgPaddingOptions::CopyBackOp::None);

  linalg::LinalgOp paddedOp;
  SmallVector<Value> newResults;
  SmallVector<tensor::PadOp> padOps;
  if (failed(rewriteAsPaddedOp(rewriter, linalgOp, options, paddedOp,
                               newResults, padOps))) {
    return rewriter.notifyMatchFailure(linalgOp,
                                       "failed to pad contraction op");
  }
  rewriter.replaceOp(linalgOp, newResults.front());
  return success();
}

struct GPUPadOperandsPass final
    : impl::GPUPadOperandsPassBase<GPUPadOperandsPass> {
  void runOnOperation() override {
    FunctionOpInterface funcOp = getOperation();

    IRRewriter rewriter(funcOp);
    funcOp.walk([&](linalg::LinalgOp op) {
      auto loweringConfig =
          getLoweringConfig<IREE::GPU::LoweringConfigAttr>(op);
      if (!loweringConfig) {
        return;
      }

      std::optional<SmallVector<int64_t>> paddingTileSizes =
          getPaddingList(loweringConfig);
      if (!paddingTileSizes) {
        return;
      }

      rewriter.setInsertionPoint(op);
      if (failed(padLinalgOpToStaticSizes(rewriter, op,
                                          paddingTileSizes.value()))) {
        return signalPassFailure();
      }
    });

    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.insert<SwapIm2ColAndPadPattern>(patterns.getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace
} // namespace mlir::iree_compiler
