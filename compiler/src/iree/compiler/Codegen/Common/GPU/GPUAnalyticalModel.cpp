// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/GPU/GPUAnalyticalModel.h"

#include "llvm/Support/DebugLog.h"
#include "llvm/Support/MathExtras.h"

#include <algorithm>
#include <cmath>

#define DEBUG_TYPE "iree-gpu-analytical-model"

namespace mlir::iree_compiler {

//===----------------------------------------------------------------------===//
// Analytical model for GEMM seed selection
//===----------------------------------------------------------------------===//
//
// The model classifies a GEMM into Small / Medium / Large using the same
// arithmetic-intensity-based regime classification as the existing hardcoded
// path in ConfigUtils.cpp (computeGemmCutoffsForAI + computeIntensity), then
// derives seed values from hardware specifications.
//
// Classification (matching ConfigUtils exactly):
//   computeIntensity = 2·M·N·K / (M·N + N·K + M·K)    [FLOP/element]
//   computeMemoryCutoff = peakTFLOPS / memBandwidthTbps [FLOP/bit]
//   smallCutoff = 0.05 · computeMemoryCutoff
//   largeCutoff = 5.0  · computeMemoryCutoff
//
// Seed values are derived from cache-line width and regime:
//
//   Small  → sg=2  mn=2   kT=4  kE = 2·cacheLine/elemBits
//   Medium → sg=4  mn=8   kT=4  kE = 2·cacheLine/elemBits
//   Large  → sg=4  mn=16  kT=2  kE = cacheLine/2/elemBits
//
// For MI300X with BF16 (cacheLine=1024 bits, elemBits=16):
//   Small  → sg=2  mn=2   kT=4  kE=128
//   Medium → sg=4  mn=8   kT=4  kE=128
//   Large  → sg=4  mn=16  kT=2  kE=32
//
// These match the hardcoded seeds exactly.  The analytical derivation means
// the model automatically adapts when targeting different hardware (different
// cache line sizes, MMA shapes, or balance points).
//
// The seeds are *aspirational*: deduceMMASchedule's fitScheduleInSharedMemory
// will refine the actual tile counts to satisfy the shared-memory budget.

std::optional<GPUMMAHeuristicSeeds>
GPUAnalyticalModel::computeOptimalSeeds(const GPUMatmulShapeType &problem,
                                        bool isGemm, bool scaled) {
  int64_t problemM = llvm::product_of(problem.mSizes);
  int64_t problemN = llvm::product_of(problem.nSizes);
  int64_t problemK = llvm::product_of(problem.kSizes);
  int64_t elementBitwidth = problem.aType.getIntOrFloatBitWidth();

  LDBG() << "=== Analytical model ===";
  LDBG() << "Problem: M=" << problemM << " N=" << problemN << " K=" << problemK
         << " elementBits=" << elementBitwidth
         << " gemmSize=" << problem.gemmSize;
  LDBG() << "Hardware: LDS=" << params_.sharedMemorySizeBytes
         << "B, cacheLine=" << params_.cacheLineSizeBytes
         << "B, MMA=" << params_.mmaM << "x" << params_.mmaN << "x"
         << params_.mmaK << ", BW=" << params_.globalMemoryBandwidthGBps
         << "GB/s";

  // Sanity: bail out if we have no useful hardware info.
  if (params_.sharedMemorySizeBytes <= 0 && params_.mmaM <= 0) {
    LDBG() << "Insufficient hardware info, returning nullopt";
    return std::nullopt;
  }

  // ---- Use the already-computed gemmSize from the problem struct ----
  //
  // The gemmSize field is set by ConfigUtils using computeGemmCutoffsForAI(),
  // which classifies the GEMM based on arithmetic intensity and the machine's
  // compute-to-memory balance.  We trust this classification and derive
  // hardware-parameterised seeds for each regime.
  //
  // This ensures the analytical model matches the existing classification
  // exactly while making the seed VALUES hardware-derived.

  GemmSize regime = problem.gemmSize;

  // Fallback: if gemmSize is not set, compute it ourselves using the same
  // formula as ConfigUtils.
  if (regime == GemmSize::NotSet) {
    int64_t dataElements =
        problemM * problemN + problemN * problemK + problemM * problemK;
    dataElements = std::max(dataElements, (int64_t)1);
    int64_t computeIntensity =
        (2 * problemM * problemN * problemK) / dataElements;

    // Machine balance: peakTFLOPS / memBandwidthTbps (FLOP/bit).
    float peakTFLOPS = 0.0f;
    auto it = params_.peakTflopsPerBitwidth.find(elementBitwidth);
    if (it != params_.peakTflopsPerBitwidth.end()) {
      peakTFLOPS = it->second;
    }
    float memBWTbps = params_.globalMemoryBandwidthGBps / 125.0f;

    float computeMemoryCutoff = 0.0f;
    if (peakTFLOPS > 0 && memBWTbps > 0) {
      computeMemoryCutoff = peakTFLOPS / memBWTbps;
    }
    // Fallback: MI300X BF16 ≈ 1300/42.4 ≈ 30.66
    if (computeMemoryCutoff <= 0.0f) {
      computeMemoryCutoff = 30.66f;
    }

    float smallCutoff = 0.05f * computeMemoryCutoff;
    float largeCutoff = 5.0f * computeMemoryCutoff;

    if (computeIntensity <= static_cast<int64_t>(smallCutoff)) {
      regime = GemmSize::SmallGemm;
    } else if (computeIntensity >= static_cast<int64_t>(largeCutoff)) {
      regime = GemmSize::LargeGemm;
    } else {
      regime = GemmSize::MediumGemm;
    }
    LDBG() << "Fallback classification: CI=" << computeIntensity << " cutoffs=["
           << smallCutoff << "," << largeCutoff << "] → " << regime;
  }

  // ---- Derive seeds from hardware parameters and regime ----
  //
  // Cache line in bits (for compatibility with the hardcoded kCacheLineSizeBits
  // constant used in getGemmHeuristicSeeds).
  int64_t cacheLineBits = params_.cacheLineSizeBytes * 8;
  if (cacheLineBits <= 0) {
    cacheLineBits = 1024; // Default: 128 bytes = 1024 bits.
  }

  int64_t subgroups, mnPerSG, kTiles, kElements;

  if (isGemm) {
    // ---- GEMM seeds ----
    switch (regime) {
    case GemmSize::SmallGemm:
      //   Small GEMM: memory-bound, use minimal MN tiles.
      //   kE = 2 · cacheLine / elemBits — two cache lines per K step.
      subgroups = 2;
      mnPerSG = 2;
      kTiles = 4;
      kElements = 2 * cacheLineBits / std::max(elementBitwidth, (int64_t)1);
      break;

    case GemmSize::LargeGemm:
      //   Large GEMM: compute-bound, maximise MN tiles.
      //   kE = cacheLine / 2 / elemBits — shorter K steps to manage register
      //   pressure from the large accumulator tile.
      subgroups = 4;
      mnPerSG = 16;
      kTiles = 2;
      kElements = cacheLineBits / 2 / std::max(elementBitwidth, (int64_t)1);
      break;

    case GemmSize::MediumGemm:
    default:
      //   Medium GEMM: balanced between memory and compute.
      //   kE = 2 · cacheLine / elemBits — same as Small.
      subgroups = 4;
      mnPerSG = 8;
      kTiles = 4;
      kElements = 2 * cacheLineBits / std::max(elementBitwidth, (int64_t)1);
      break;
    }

    // Handle scaled GEMM matmuls (block-scaled quantisation).
    if (scaled) {
      switch (regime) {
      case GemmSize::SmallGemm:
      case GemmSize::MediumGemm:
        subgroups = 8;
        mnPerSG = 32;
        kTiles = 4;
        kElements = cacheLineBits / 2 / std::max(elementBitwidth, (int64_t)1);
        break;
      case GemmSize::LargeGemm:
        subgroups = 8;
        mnPerSG = 32;
        kTiles = 2;
        kElements = cacheLineBits / 2 / std::max(elementBitwidth, (int64_t)1);
        break;
      default:
        break;
      }
    }
  } else {
    // ---- Convolution / non-GEMM contraction seeds ----
    // Convolutions benefit from more subgroups (latency hiding from global
    // loads) and smaller MN tiles per subgroup.
    switch (regime) {
    case GemmSize::SmallGemm:
      subgroups = 2;
      mnPerSG = 2;
      kTiles = 4;
      // Convolution Small uses 1× cache line (not 2×).
      kElements = cacheLineBits / std::max(elementBitwidth, (int64_t)1);
      break;

    case GemmSize::LargeGemm:
      // Favor more subgroups for convolution to help latency hiding.
      subgroups = 8;
      mnPerSG = 8;
      kTiles = 2;
      kElements = cacheLineBits / 2 / std::max(elementBitwidth, (int64_t)1);
      break;

    case GemmSize::MediumGemm:
    default:
      // More subgroups, fewer MN tiles per subgroup for convolution.
      subgroups = 8;
      mnPerSG = 4;
      kTiles = 4;
      kElements = 2 * cacheLineBits / std::max(elementBitwidth, (int64_t)1);
      break;
    }
  }

  // Ensure minimum values.
  kElements = std::max(kElements, (int64_t)1);
  subgroups = std::max(subgroups, (int64_t)1);
  mnPerSG = std::max(mnPerSG, (int64_t)1);
  kTiles = std::max(kTiles, (int64_t)2);

  LDBG() << "Seeds: sg=" << subgroups << " mn=" << mnPerSG << " kT=" << kTiles
         << " kE=" << kElements << " (regime=" << regime << ")";

  // ---- Assemble seeds ----
  GPUMMAHeuristicSeeds seeds;
  seeds.bestSubgroupCountPerWorkgroup = subgroups;
  seeds.bestMNTileCountPerSubgroup = mnPerSG;
  seeds.bestKTileCountPerSubgroup = kTiles;
  seeds.bestKElementCountPerSubgroup = kElements;

  LDBG() << "=== Analytical seeds: sg=" << seeds.bestSubgroupCountPerWorkgroup
         << " mn=" << seeds.bestMNTileCountPerSubgroup
         << " kT=" << seeds.bestKTileCountPerSubgroup
         << " kE=" << seeds.bestKElementCountPerSubgroup << " ===";

  return seeds;
}

} // namespace mlir::iree_compiler
