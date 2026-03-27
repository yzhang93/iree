// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Preprocessing/Common/Passes.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"

namespace mlir::iree_compiler::Preprocessing {

#define GEN_PASS_DEF_CONVERTSTRIDEDINSERTSLICETOGENERICPASS
#include "iree/compiler/Preprocessing/Common/Passes.h.inc"

namespace {

/// Convert a strided `tensor.insert_slice` into a zero-constant destination
/// to a `linalg.generic` with index arithmetic.
///
/// This targets the backward-data convolution pattern where the upstream
/// gradient is scattered into a zero buffer at strided positions, producing
/// a separate Memset + slow_memcpy dispatch pair. The replacement generic
/// computes the strided scatter in a single pass and is potentially
/// fusable with consumer ops in later dispatch formation passes.
///
/// For each output position, the generic checks whether the position maps
/// to a valid source element (i.e., (pos - offset) is non-negative,
/// divisible by stride, and the quotient is in-bounds). Source indices are
/// clamped to valid range so the extract is always safe, and arith.select
/// chooses between the extracted value and zero. This avoids scf.if
/// branches that prevent vectorization and cause GPU branch divergence.
/// Power-of-2 strides use bitwise ops instead of expensive integer division.
class ConvertStridedInsertSliceToGeneric
    : public OpRewritePattern<tensor::InsertSliceOp> {
public:
  using Base::Base;
  LogicalResult matchAndRewrite(tensor::InsertSliceOp op,
                                PatternRewriter &rewriter) const override {
    // Destination must be a zero splat constant.
    Value dest = op.getDest();
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

    // All offsets, sizes, and strides must be static, with at least one
    // non-unit stride.
    SmallVector<int64_t> offsets(op.getStaticOffsets());
    SmallVector<int64_t> strides(op.getStaticStrides());
    SmallVector<int64_t> sizes(op.getStaticSizes());
    if (ShapedType::isDynamicShape(offsets) ||
        ShapedType::isDynamicShape(strides) ||
        ShapedType::isDynamicShape(sizes)) {
      return failure();
    }
    if (llvm::all_of(strides, [](int64_t s) { return s == 1; })) {
      return failure();
    }

    Value src = op.getSource();
    auto srcTy = cast<RankedTensorType>(src.getType());
    auto destTy = cast<RankedTensorType>(dest.getType());
    unsigned rank = destTy.getRank();
    auto elemTy = destTy.getElementType();
    Location loc = op.getLoc();

    Value empty =
        tensor::EmptyOp::create(rewriter, loc, destTy.getShape(), elemTy);
    AffineMap identityMap = rewriter.getMultiDimIdentityMap(rank);
    SmallVector<AffineMap> indexingMaps = {identityMap};
    SmallVector<utils::IteratorType> iterTypes(rank,
                                               utils::IteratorType::parallel);

    auto genericOp = linalg::GenericOp::create(
        rewriter, loc, destTy, /*inputs=*/ValueRange{},
        /*outputs=*/ValueRange{empty}, indexingMaps, iterTypes,
        [&](OpBuilder &b, Location loc, ValueRange args) {
          Value allValid = nullptr;
          SmallVector<Value> srcIndices;
          Value zero = arith::ConstantOp::create(b, loc, b.getIndexAttr(0));
          for (unsigned i = 0; i < rank; i++) {
            Value idx = linalg::IndexOp::create(b, loc, i);

            if (strides[i] == 1 && offsets[i] == 0) {
              srcIndices.push_back(idx);
              continue;
            }

            Value off =
                arith::ConstantOp::create(b, loc, b.getIndexAttr(offsets[i]));
            Value srcSize = arith::ConstantOp::create(
                b, loc, b.getIndexAttr(srcTy.getDimSize(i)));

            Value shifted = arith::SubIOp::create(b, loc, idx, off);

            Value rem, srcIdx;
            bool isPow2 = (strides[i] & (strides[i] - 1)) == 0;
            if (isPow2) {
              Value strideMask = arith::ConstantOp::create(
                  b, loc, b.getIndexAttr(strides[i] - 1));
              Value log2Stride = arith::ConstantOp::create(
                  b, loc, b.getIndexAttr(llvm::Log2_64(strides[i])));
              rem = arith::AndIOp::create(b, loc, shifted, strideMask);
              srcIdx = arith::ShRSIOp::create(b, loc, shifted, log2Stride);
            } else {
              Value str =
                  arith::ConstantOp::create(b, loc, b.getIndexAttr(strides[i]));
              rem = arith::RemSIOp::create(b, loc, shifted, str);
              srcIdx = arith::DivSIOp::create(b, loc, shifted, str);
            }

            Value geZero = arith::CmpIOp::create(
                b, loc, arith::CmpIPredicate::sge, shifted, zero);
            Value remZero = arith::CmpIOp::create(
                b, loc, arith::CmpIPredicate::eq, rem, zero);
            Value ltSize = arith::CmpIOp::create(
                b, loc, arith::CmpIPredicate::slt, srcIdx, srcSize);
            Value dimValid = arith::AndIOp::create(b, loc, geZero, remZero);
            dimValid = arith::AndIOp::create(b, loc, dimValid, ltSize);

            allValid = allValid
                           ? arith::AndIOp::create(b, loc, allValid, dimValid)
                           : dimValid;

            // Clamp to [0, srcSize-1] so the extract is always in-bounds,
            // even when the predicate is false.
            Value srcSizeM1 = arith::SubIOp::create(
                b, loc, srcSize,
                arith::ConstantOp::create(b, loc, b.getIndexAttr(1)));
            Value clamped = arith::MaxSIOp::create(b, loc, srcIdx, zero);
            clamped = arith::MinSIOp::create(b, loc, clamped, srcSizeM1);
            srcIndices.push_back(clamped);
          }

          Value zeroElem;
          if (isa<FloatType>(elemTy)) {
            zeroElem =
                arith::ConstantOp::create(b, loc, b.getFloatAttr(elemTy, 0.0));
          } else {
            zeroElem =
                arith::ConstantOp::create(b, loc, b.getIntegerAttr(elemTy, 0));
          }

          // Always extract (indices are clamped to valid range).
          Value extracted = tensor::ExtractOp::create(b, loc, src, srcIndices);

          if (!allValid) {
            // No strided dims had predicates; just yield the extracted value.
            linalg::YieldOp::create(b, loc, extracted);
            return;
          }

          // Select between extracted value and zero based on predicate.
          Value result =
              arith::SelectOp::create(b, loc, allValid, extracted, zeroElem);
          linalg::YieldOp::create(b, loc, result);
        });

    rewriter.replaceOp(op, genericOp.getResult(0));
    return success();
  }
};

struct ConvertStridedInsertSliceToGenericPass
    : impl::ConvertStridedInsertSliceToGenericPassBase<
          ConvertStridedInsertSliceToGenericPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, linalg::LinalgDialect,
                    tensor::TensorDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.insert<ConvertStridedInsertSliceToGeneric>(context);
    walkAndApplyPatterns(getOperation(), std::move(patterns));
  }
};

} // namespace
} // namespace mlir::iree_compiler::Preprocessing
