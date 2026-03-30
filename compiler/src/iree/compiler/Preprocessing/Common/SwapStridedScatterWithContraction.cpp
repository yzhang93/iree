// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Preprocessing/Common/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"

namespace mlir::iree_compiler::Preprocessing {

#define GEN_PASS_DEF_SWAPSTRIDEDSCATTERWITHCONTRACTIONPASS
#include "iree/compiler/Preprocessing/Common/Passes.h.inc"

namespace {

/// Swap a strided scatter (tensor.insert_slice into zeros) with its consumer
/// contraction (matmul/conv) when the contraction reads the scattered tensor
/// with a projected permutation on the strided dims. This is valid for 1x1
/// backward convolutions where the scatter commutes with the matmul.
///
/// Before:
///   %scattered = insert_slice %src into %zeros [offs][sizes][strides]
///   %result = contraction(%scattered, %filter)
///   %trunced = truncf(%result)
///
/// After:
///   %small_result = contraction(%src, %filter)
///   %small_trunced = truncf(%small_result)
///   %result = insert_slice %small_trunced into %zeros' [offs][sizes'][strides]
class SwapStridedScatterWithContraction
    : public OpRewritePattern<tensor::InsertSliceOp> {
public:
  using Base::Base;
  LogicalResult matchAndRewrite(tensor::InsertSliceOp insertOp,
                                PatternRewriter &rewriter) const override {
    // Destination must be a zero splat constant.
    Value dest = insertOp.getDest();
    Attribute destAttr;
    if (!matchPattern(dest, m_Constant(&destAttr))) {
      return failure();
    }
    auto splatAttr = dyn_cast<SplatElementsAttr>(destAttr);
    if (!splatAttr) {
      return failure();
    }
    Attribute splatVal = splatAttr.getSplatValue<Attribute>();
    bool isZero =
        TypeSwitch<Attribute, bool>(splatVal)
            .Case<FloatAttr>([](auto a) { return a.getValue().isZero(); })
            .Case<IntegerAttr>([](auto a) { return a.getValue().isZero(); })
            .Default([](auto) { return false; });
    if (!isZero) {
      return failure();
    }

    // Must have static metadata with at least one non-unit stride.
    SmallVector<int64_t> offsets(insertOp.getStaticOffsets());
    SmallVector<int64_t> strides(insertOp.getStaticStrides());
    SmallVector<int64_t> sizes(insertOp.getStaticSizes());
    if (ShapedType::isDynamicShape(offsets) ||
        ShapedType::isDynamicShape(strides) ||
        ShapedType::isDynamicShape(sizes)) {
      return failure();
    }
    if (llvm::all_of(strides, [](int64_t s) { return s == 1; })) {
      return failure();
    }

    // The insert_slice result must feed into a contraction, possibly through
    // a compute_barrier.
    if (!insertOp->hasOneUse()) {
      return failure();
    }
    Value scatterResult = insertOp.getResult();
    Operation *user = *insertOp->user_begin();
    // Look through compute_barrier.start if present.
    if (user->getName().getStringRef() ==
        "iree_tensor_ext.compute_barrier.start") {
      if (!user->hasOneUse()) {
        return failure();
      }
      scatterResult = user->getResult(0);
      user = *user->user_begin();
    }
    auto convOp = dyn_cast<linalg::GenericOp>(user);
    if (!convOp) {
      return failure();
    }

    // Must be a contraction: 2 inputs, 1 init, has reductions.
    if (convOp.getNumDpsInputs() != 2 || convOp.getNumDpsInits() != 1) {
      return failure();
    }
    if (convOp.getNumReductionLoops() == 0) {
      return failure();
    }

    // Check which operand is the scattered tensor.
    int scatterOperandIdx = -1;
    if (convOp.getDpsInputOperand(0)->get() == scatterResult) {
      scatterOperandIdx = 0;
    } else if (convOp.getDpsInputOperand(1)->get() == scatterResult) {
      scatterOperandIdx = 1;
    } else {
      return failure();
    }

    // For each strided input dim, the contraction's indexing map must
    // reference exactly one parallel dim (possibly summed with reduction
    // dims that have loop bound 1, as in 1x1 convolutions where the map
    // is `d_spatial + d_kernel` with kernel size 1).
    AffineMap scatterMap = convOp.getIndexingMapsArray()[scatterOperandIdx];
    SmallVector<int64_t> loopRanges =
        cast<linalg::LinalgOp>(convOp.getOperation()).getStaticLoopRanges();

    // For strided dims, verify each map expression resolves to a single
    // parallel dim (plus optional unit-range reduction dims).
    for (unsigned d = 0; d < scatterMap.getNumResults(); d++) {
      if (strides[d] == 1) {
        continue;
      }
      AffineExpr expr = scatterMap.getResult(d);
      // Collect all dim positions in this expression.
      SmallVector<unsigned> dimPositions;
      expr.walk([&](AffineExpr e) {
        if (auto dim = dyn_cast<AffineDimExpr>(e)) {
          dimPositions.push_back(dim.getPosition());
        }
      });
      if (dimPositions.empty()) {
        return failure();
      }
      // Exactly one must be a parallel dim; the rest must be reduction
      // dims with loop bound 1.
      int parallelCount = 0;
      auto iterTypes = convOp.getIteratorTypesArray();
      for (unsigned pos : dimPositions) {
        if (iterTypes[pos] == utils::IteratorType::parallel) {
          parallelCount++;
        } else {
          if (pos >= loopRanges.size() || loopRanges[pos] != 1) {
            return failure();
          }
        }
      }
      if (parallelCount != 1) {
        return failure();
      }
    }

    // Check contraction body.
    if (!mlir::linalg::detail::isContractionBody(
            *convOp.getBlock(), [](Operation *first, Operation *second) {
              return (isa<arith::MulFOp>(first) &&
                      isa<arith::AddFOp>(second)) ||
                     (isa<arith::MulIOp>(first) && isa<arith::AddIOp>(second));
            })) {
      return failure();
    }

    Value src = insertOp.getSource();
    auto srcTy = cast<RankedTensorType>(src.getType());
    auto scatteredTy = cast<RankedTensorType>(insertOp.getResult().getType());
    auto resultTy = cast<RankedTensorType>(convOp.getResultTypes()[0]);
    AffineMap resultMap = convOp.getIndexingMapsArray().back();
    unsigned rank = scatteredTy.getRank();
    Location loc = insertOp.getLoc();

    // Compute the new (smaller) result shape: replace the scattered spatial
    // dims with the source spatial dims.
    auto iterTypes = convOp.getIteratorTypesArray();
    SmallVector<int64_t> newResultShape(resultTy.getShape());
    for (unsigned d = 0; d < rank; d++) {
      if (strides[d] == 1) {
        continue;
      }
      // Find the parallel contraction dim that reads this input dim.
      // The expression may be `d_parallel + d_reduction` for 1x1 convs.
      AffineExpr scatterExpr = scatterMap.getResult(d);
      int parallelDim = -1;
      scatterExpr.walk([&](AffineExpr e) {
        if (auto dim = dyn_cast<AffineDimExpr>(e)) {
          if (iterTypes[dim.getPosition()] == utils::IteratorType::parallel) {
            parallelDim = dim.getPosition();
          }
        }
      });
      if (parallelDim < 0) {
        return failure();
      }
      unsigned contractionDim = parallelDim;
      // Find which result dim this contraction dim maps to.
      for (auto [resIdx, resExpr] : llvm::enumerate(resultMap.getResults())) {
        auto resDimExpr = dyn_cast<AffineDimExpr>(resExpr);
        if (resDimExpr && resDimExpr.getPosition() == contractionDim) {
          newResultShape[resIdx] = srcTy.getDimSize(d);
        }
      }
    }

    auto newResultTy =
        RankedTensorType::get(newResultShape, resultTy.getElementType());

    // Create the fill for the smaller result.
    // Set insertion point before the conv, which dominates all its operands.
    rewriter.setInsertionPoint(convOp);
    auto fillOp =
        convOp.getDpsInitOperand(0)->get().getDefiningOp<linalg::FillOp>();
    if (!fillOp) {
      return failure();
    }
    Value newEmpty = tensor::EmptyOp::create(rewriter, loc, newResultShape,
                                             resultTy.getElementType());
    Value newFill = linalg::FillOp::create(rewriter, loc, fillOp.getInputs(),
                                           ValueRange{newEmpty})
                        .getResult(0);

    // Create the smaller contraction.
    Value otherInput =
        convOp.getDpsInputOperand(scatterOperandIdx == 0 ? 1 : 0)->get();
    SmallVector<Value> newInputs;
    if (scatterOperandIdx == 0) {
      newInputs = {src, otherInput};
    } else {
      newInputs = {otherInput, src};
    }

    auto newConv = linalg::GenericOp::create(
        rewriter, loc, newResultTy, newInputs, ValueRange{newFill},
        convOp.getIndexingMapsArray(), convOp.getIteratorTypesArray(),
        [&](OpBuilder &b, Location l, ValueRange args) {
          IRMapping mapping;
          for (auto [oldArg, newArg] :
               llvm::zip(convOp.getBlock()->getArguments(), args)) {
            mapping.map(oldArg, newArg);
          }
          for (auto &op : convOp.getBlock()->without_terminator()) {
            b.clone(op, mapping);
          }
          auto yield =
              cast<linalg::YieldOp>(convOp.getBlock()->getTerminator());
          SmallVector<Value> yieldOps;
          for (Value v : yield.getOperands()) {
            yieldOps.push_back(mapping.lookupOrDefault(v));
          }
          linalg::YieldOp::create(b, l, yieldOps);
        });

    // Walk the conv's users to find the truncf consumer chain, then apply
    // the insert_slice after the last consumer.
    Value currentResult = newConv.getResult(0);

    // Collect the chain of elementwise consumers (truncf, etc.) and clone
    // them for the smaller result, then insert_slice.
    SmallVector<Operation *> consumerChain;
    Operation *lastOp = convOp;
    while (lastOp->hasOneUse()) {
      Operation *user = *lastOp->user_begin();
      auto genericUser = dyn_cast<linalg::GenericOp>(user);
      if (!genericUser) {
        break;
      }
      if (genericUser.getNumDpsInputs() != 1 ||
          genericUser.getNumDpsInits() != 1) {
        break;
      }
      if (genericUser.getNumReductionLoops() != 0) {
        break;
      }
      consumerChain.push_back(genericUser);
      lastOp = genericUser;
    }

    // Clone the consumer chain for the smaller result.
    for (Operation *consumer : consumerChain) {
      auto genericConsumer = cast<linalg::GenericOp>(consumer);
      auto consumerResultTy =
          cast<RankedTensorType>(genericConsumer.getResultTypes()[0]);
      auto newConsumerResultTy = RankedTensorType::get(
          newResultShape, consumerResultTy.getElementType());
      Value consumerEmpty = tensor::EmptyOp::create(
          rewriter, loc, newResultShape, consumerResultTy.getElementType());
      auto newConsumer = linalg::GenericOp::create(
          rewriter, loc, newConsumerResultTy, ValueRange{currentResult},
          ValueRange{consumerEmpty}, genericConsumer.getIndexingMapsArray(),
          genericConsumer.getIteratorTypesArray(),
          [&](OpBuilder &b, Location l, ValueRange args) {
            IRMapping mapping;
            for (auto [oldArg, newArg] :
                 llvm::zip(genericConsumer.getBlock()->getArguments(), args)) {
              mapping.map(oldArg, newArg);
            }
            for (auto &op : genericConsumer.getBlock()->without_terminator()) {
              b.clone(op, mapping);
            }
            auto yield = cast<linalg::YieldOp>(
                genericConsumer.getBlock()->getTerminator());
            SmallVector<Value> yieldOps;
            for (Value v : yield.getOperands()) {
              yieldOps.push_back(mapping.lookupOrDefault(v));
            }
            linalg::YieldOp::create(b, l, yieldOps);
          });
      currentResult = newConsumer.getResult(0);
    }

    // Create the output scatter: fill + insert_slice.
    // The output shape is the original conv result shape (large spatial dims)
    // but with the element type from the consumer chain (e.g., bf16 after
    // truncf).
    auto outputElemTy =
        cast<RankedTensorType>(currentResult.getType()).getElementType();
    // Map the result dims back to the scattered tensor layout.
    // For strided dims: keep the large (scattered) spatial size.
    // For non-strided dims: use the result's dim size.
    SmallVector<int64_t> outputShape(rank);
    auto smallResultShape =
        cast<RankedTensorType>(currentResult.getType()).getShape();
    for (unsigned d = 0; d < rank; d++) {
      if (strides[d] != 1) {
        // Strided dim: keep the large scattered spatial size.
        outputShape[d] = scatteredTy.getDimSize(d);
      } else {
        // Non-strided dim: use the small result's corresponding dim.
        outputShape[d] = smallResultShape[d];
      }
    }

    auto outputTy = RankedTensorType::get(outputShape, outputElemTy);
    Value outputZeros = arith::ConstantOp::create(
        rewriter, loc,
        SplatElementsAttr::get(outputTy, rewriter.getZeroAttr(outputElemTy)));

    SmallVector<int64_t> newSizes(newResultShape);
    auto newInsertSlice = tensor::InsertSliceOp::create(
        rewriter, loc, currentResult, outputZeros,
        getAsOpFoldResult(rewriter.getI64ArrayAttr(offsets)),
        getAsOpFoldResult(rewriter.getI64ArrayAttr(newSizes)),
        getAsOpFoldResult(rewriter.getI64ArrayAttr(strides)));

    // Replace the last consumer (or the conv itself) with the new result.
    if (consumerChain.empty()) {
      rewriter.replaceOp(convOp, newInsertSlice.getResult());
    } else {
      rewriter.replaceOp(consumerChain.back(), newInsertSlice.getResult());
      // Erase the old consumer chain and conv.
      for (auto it = consumerChain.rbegin() + 1; it != consumerChain.rend();
           ++it) {
        rewriter.eraseOp(*it);
      }
      rewriter.eraseOp(convOp);
    }
    rewriter.eraseOp(insertOp);

    return success();
  }
};

struct SwapStridedScatterWithContractionPass
    : impl::SwapStridedScatterWithContractionPassBase<
          SwapStridedScatterWithContractionPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, linalg::LinalgDialect,
                    tensor::TensorDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.insert<SwapStridedScatterWithContraction>(context);
    walkAndApplyPatterns(getOperation(), std::move(patterns));
  }
};

} // namespace
} // namespace mlir::iree_compiler::Preprocessing
