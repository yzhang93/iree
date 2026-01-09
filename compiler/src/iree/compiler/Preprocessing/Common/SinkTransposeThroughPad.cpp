// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Flow/Transforms/RegionOpUtils.h"
#include "iree/compiler/Preprocessing/Common/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::Preprocessing {

#define GEN_PASS_DEF_SINKTRANSPOSETHROUGHPADPASS
#include "iree/compiler/Preprocessing/Common/Passes.h.inc"

static Value createTransposeInit(OpBuilder &builder, Value source,
                                 ArrayRef<int64_t> perm) {
  SmallVector<OpFoldResult> mixedSizes =
      tensor::getMixedSizes(builder, source.getLoc(), source);
  applyPermutationToVector(mixedSizes, perm);
  Type elemType = cast<RankedTensorType>(source.getType()).getElementType();
  Value empty =
      tensor::EmptyOp::create(builder, source.getLoc(), mixedSizes, elemType)
          .getResult();
  return empty;
}

static Value createTranspose(OpBuilder &builder, Value source,
                             ArrayRef<int64_t> perm) {
  Value empty = createTransposeInit(builder, source, perm);
  return linalg::TransposeOp::create(builder, source.getLoc(), source, empty,
                                     perm)
      ->getResult(0);
}

// Sinks a transpose through a tensor.expand_shape
// Adapted from PropagateLinalgTranspose.cpp::SinkTransposeThroughExpandShape
class SinkTransposeThroughExpandShapeOp
    : public OpRewritePattern<tensor::ExpandShapeOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::ExpandShapeOp expandOp,
                                PatternRewriter &rewriter) const override {
    if (!IREE::Flow::isNonNullAndOutsideDispatch(expandOp)) {
      return failure();
    }
    Value source = expandOp.getSrc();
    auto transposeOp = source.getDefiningOp<linalg::TransposeOp>();
    if (!transposeOp) {
      return failure();
    }

    // Get the inverse permutation
    auto invPerm = invertPermutationVector(transposeOp.getPermutation());
    SmallVector<ReassociationIndices> reassociations =
        expandOp.getReassociationIndices();

    // Because we are doing expand_shape(transpose), all expanded groups are
    // transposed together. As a result, to get the permutation of the new
    // transpose, we can just flatten the transposed reassociation indices.
    // For example:
    //   permutation = [0, 2, 1]
    //   reassociation_map = [[0, 1, 2], [3], [4, 5]]
    // Becomes:
    //   reassociation_map = [[0, 1, 2], [3, 4], [5]]
    //   permutation = [0, 1, 2, 4, 5, 3]
    applyPermutationToVector(reassociations, invPerm);

    SmallVector<int64_t> newInvPerm;
    SmallVector<ReassociationIndices> newReassociations;
    int64_t expandedDim = 0;
    for (auto reassoc : reassociations) {
      ReassociationIndices newReassoc;
      for (auto dim : reassoc) {
        newInvPerm.push_back(dim);
        newReassoc.push_back(expandedDim++);
      }
      newReassociations.push_back(newReassoc);
    }

    auto newPerm = invertPermutationVector(newInvPerm);

    // Compute the new expanded type by permuting the shape
    auto oldExpandedType = cast<RankedTensorType>(expandOp.getType());
    SmallVector<int64_t> newExpandedShape;
    for (int64_t dim : newInvPerm) {
      newExpandedShape.push_back(oldExpandedType.getShape()[dim]);
    }
    auto newExpandedType = RankedTensorType::get(
        newExpandedShape, oldExpandedType.getElementType(),
        oldExpandedType.getEncoding());

    // Create new expand_shape on the untransposed input
    Value transposedReshape = tensor::ExpandShapeOp::create(
        rewriter, expandOp.getLoc(), newExpandedType, transposeOp.getInput(),
        newReassociations);

    // Create new transpose on the expanded result
    Value originalReshape =
        createTranspose(rewriter, transposedReshape, newPerm);
    rewriter.replaceOp(expandOp, originalReshape);
    return success();
  }
};

// Sinks a transpose through a tensor.pad
class SinkTransposeThroughPadOp : public OpRewritePattern<tensor::PadOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::PadOp padOp,
                                PatternRewriter &rewriter) const override {
    if (!IREE::Flow::isNonNullAndOutsideDispatch(padOp)) {
      return failure();
    }
    Value source = padOp.getSource();
    auto transposeOp = source.getDefiningOp<linalg::TransposeOp>();
    if (!transposeOp) {
      return failure();
    }

    Block &block = padOp.getRegion().front();
    if (llvm::any_of(block.getArguments(), [](BlockArgument blockArg) {
          return blockArg.getNumUses();
        })) {
      return failure();
    }

    auto invPerm = invertPermutationVector(transposeOp.getPermutation());
    SmallVector<OpFoldResult> lowSizes = padOp.getMixedLowPad();
    SmallVector<OpFoldResult> highSizes = padOp.getMixedHighPad();
    applyPermutationToVector(lowSizes, invPerm);
    applyPermutationToVector(highSizes, invPerm);

    RankedTensorType oldPaddedType = cast<RankedTensorType>(padOp.getType());
    RankedTensorType newPaddedType = oldPaddedType.clone(
        applyPermutation(oldPaddedType.getShape(), invPerm));
    auto newPadOp = tensor::PadOp::create(
        rewriter, padOp.getLoc(), newPaddedType, transposeOp.getInput(),
        lowSizes, highSizes, padOp.getNofold());
    rewriter.cloneRegionBefore(padOp.getRegion(), newPadOp.getRegion(),
                               newPadOp.getRegion().begin());
    Value newTransposeOp =
        createTranspose(rewriter, newPadOp, transposeOp.getPermutation());
    rewriter.replaceOp(padOp, newTransposeOp);
    return success();
  }
};

namespace {
struct SinkTransposeThroughPadPass
    : public impl::SinkTransposeThroughPadPassBase<
          SinkTransposeThroughPadPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns
        .insert<SinkTransposeThroughExpandShapeOp, SinkTransposeThroughPadOp>(
            &getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      getOperation().emitError(getPassName()) << " failed to converge.";
      return signalPassFailure();
    }
  }
};
} // namespace

} // namespace mlir::iree_compiler::Preprocessing
