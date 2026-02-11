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
// 1.  BLOCK_K — derived from cache-line width
//===----------------------------------------------------------------------===//
// The narrowest useful K-tile is one that completely fills a cache line so that
// every global-memory transaction is fully utilised.
//
//   BLOCK_K = cacheLineBytes / elementBytes
//
// This is exactly the rule tritonBLAS uses.  For FP16 with a 128-byte cache
// line this gives BLOCK_K = 64.

int64_t GPUAnalyticalModel::deriveBlockK(int64_t elementBytes) const {
  if (elementBytes <= 0) {
    elementBytes = 2; // FP16 fallback
  }
  int64_t blockK = params_.cacheLineSizeBytes / elementBytes;
  // Must be at least as large as the MMA intrinsic K dimension.
  blockK = std::max(blockK, params_.mmaK);
  // Round up to the nearest multiple of the intrinsic K so that the
  // downstream schedule can tile evenly.
  if (params_.mmaK > 0) {
    blockK = llvm::alignTo(blockK, params_.mmaK);
  }
  return blockK;
}

//===----------------------------------------------------------------------===//
// 2.  Shared-memory footprint modelling
//===----------------------------------------------------------------------===//
// For one pipeline stage the LDS stores the A-tile (blockM × blockK) and the
// B-tile (blockN × blockK).

int64_t GPUAnalyticalModel::sharedMemPerStage(int64_t blockM, int64_t blockN,
                                              int64_t blockK,
                                              int64_t elementBytes) const {
  return (blockM * blockK + blockN * blockK) * elementBytes;
}

//===----------------------------------------------------------------------===//
// 3.  Pipeline depth (number of K-tiles buffered in LDS)
//===----------------------------------------------------------------------===//
// stages = floor(sharedMemorySize / sharedMemPerStage)
// Clamped to [2, 8] — at least double-buffered, but not so deep that the
// working set evicts useful data from the L1/L2 hierarchy.

int64_t GPUAnalyticalModel::deriveMaxPipelineDepth(int64_t blockM,
                                                   int64_t blockN,
                                                   int64_t blockK,
                                                   int64_t elementBytes) const {
  int64_t perStage = sharedMemPerStage(blockM, blockN, blockK, elementBytes);
  if (perStage <= 0 || params_.sharedMemorySizeBytes <= 0) {
    return 2; // double-buffer minimum
  }
  int64_t stages = params_.sharedMemorySizeBytes / perStage;
  return std::max(static_cast<int64_t>(2),
                  std::min(stages, static_cast<int64_t>(8)));
}

//===----------------------------------------------------------------------===//
// 4.  Maximum square-ish MN block that fits in LDS
//===----------------------------------------------------------------------===//
// Solve for blockMN (assuming square tiles, blockM ≈ blockN ≈ blockMN):
//
//   (blockMN + blockMN) * blockK * elementBytes * stages ≤ LDS
//   blockMN ≤ LDS / (2 * blockK * elementBytes * stages)
//
// We round down to the nearest multiple of the MMA intrinsic M/N dimension.

int64_t GPUAnalyticalModel::deriveMaxBlockMN(int64_t blockK,
                                             int64_t elementBytes,
                                             int64_t pipelineDepth) const {
  if (blockK <= 0 || elementBytes <= 0 || pipelineDepth <= 0 ||
      params_.sharedMemorySizeBytes <= 0) {
    return params_.mmaM > 0 ? params_.mmaM * 4 : 64;
  }
  int64_t maxBlockMN = params_.sharedMemorySizeBytes /
                       (2 * blockK * elementBytes * pipelineDepth);
  // Round down to the nearest multiple of the intrinsic.
  int64_t intrinsicMN = std::max(params_.mmaM, params_.mmaN);
  if (intrinsicMN > 0 && maxBlockMN >= intrinsicMN) {
    maxBlockMN = (maxBlockMN / intrinsicMN) * intrinsicMN;
  }
  // At minimum, one intrinsic tile.
  return std::max(maxBlockMN,
                  intrinsicMN > 0 ? intrinsicMN : static_cast<int64_t>(16));
}

//===----------------------------------------------------------------------===//
// 5.  Register-pressure estimate
//===----------------------------------------------------------------------===//
// Each subgroup holds:
//   - Accumulator:   (mPerSG * nPerSG) elements  (full precision, 32-bit)
//   - A fragment:    (mPerSG * kIntrinsic)  elements
//   - B fragment:    (nPerSG * kIntrinsic)  elements
//   - Overhead:      ~16 VGPRs for indices / predicates / addresses
//
// Total VGPRs ≈ (accum_elements * 4 + frag_elements * elementBytes) / 4 + 16
// (assuming 4 bytes per VGPR).

int64_t
GPUAnalyticalModel::estimateVgprsPerSubgroup(int64_t mTilesPerSubgroup,
                                             int64_t nTilesPerSubgroup,
                                             int64_t elementBytes) const {
  int64_t mPerSG = mTilesPerSubgroup * params_.mmaM;
  int64_t nPerSG = nTilesPerSubgroup * params_.mmaN;

  // Accumulator elements are typically stored in FP32 (4 bytes each),
  // distributed across the subgroup.
  int64_t accumElements = mPerSG * nPerSG;
  int64_t accumVgprs = (accumElements * 4) / (4 * params_.subgroupSize);

  // Input fragments for one K-step (one MMA intrinsic invocation).
  int64_t fragA = mPerSG * params_.mmaK;
  int64_t fragB = nPerSG * params_.mmaK;
  int64_t fragVgprs =
      ((fragA + fragB) * elementBytes) / (4 * params_.subgroupSize);

  int64_t overhead = 16;
  return accumVgprs + fragVgprs + overhead;
}

//===----------------------------------------------------------------------===//
// 6.  Subgroup count from occupancy model
//===----------------------------------------------------------------------===//
// The GPU can run at most `maxWavesPerSimd` waves per SIMD.  Each workgroup
// occupies `subgroupCount` waves.  The number of concurrent workgroups per
// SIMD is:
//
//   wg_per_simd = floor(maxWavesPerSimd / subgroupCount)
//
// Occupancy is limited by three resources:
//   (a) Waves:   subgroupCount ≤ maxWavesPerSimd
//   (b) VGPRs:   subgroupCount * vgprs_per_sg ≤ maxVgprsPerSimd
//   (c) LDS:     shared_mem_per_wg ≤ sharedMemorySize
//
// We pick the *largest* subgroup count that satisfies all three, because
// more subgroups ⇒ more work per workgroup ⇒ less dispatch overhead.

int64_t GPUAnalyticalModel::deriveSubgroupCount(int64_t blockM, int64_t blockN,
                                                int64_t blockK,
                                                int64_t elementBytes,
                                                int64_t pipelineDepth) const {
  if (params_.mmaM <= 0 || params_.mmaN <= 0) {
    return 4; // Safe default when intrinsic info is missing
  }

  // Start from the hardware maximum and work downward.
  int64_t maxByWaves = params_.maxWavesPerSimd;

  // We want at least 2 workgroups per SIMD for latency hiding, so limit
  // subgroup count so that 2 WGs can coexist:
  //   subgroupCount ≤ maxWavesPerSimd / 2
  int64_t targetConcurrentWGs = 2;
  int64_t maxForConcurrency = maxByWaves / targetConcurrentWGs;
  maxForConcurrency = std::max(maxForConcurrency, static_cast<int64_t>(1));

  // Try subgroup counts from maxForConcurrency down to 1 and pick the
  // first that fits in the VGPR budget.
  for (int64_t sg = maxForConcurrency; sg >= 1; --sg) {
    // Distribute blockM × blockN across `sg` subgroups.
    // Simplification: assume roughly equal split along the larger dimension.
    int64_t mTilesTotal =
        std::max(blockM / params_.mmaM, static_cast<int64_t>(1));
    int64_t nTilesTotal =
        std::max(blockN / params_.mmaN, static_cast<int64_t>(1));
    int64_t totalTiles = mTilesTotal * nTilesTotal;
    int64_t tilesPerSG = std::max(totalTiles / sg, static_cast<int64_t>(1));

    // Approximate split: sqrt(tilesPerSG) along each dimension.
    int64_t sqrtTiles =
        static_cast<int64_t>(std::sqrt(static_cast<double>(tilesPerSG)));
    sqrtTiles = std::max(sqrtTiles, static_cast<int64_t>(1));
    int64_t mTilesPerSG = sqrtTiles;
    int64_t nTilesPerSG =
        std::max(tilesPerSG / sqrtTiles, static_cast<int64_t>(1));

    int64_t vgprsPerSG =
        estimateVgprsPerSubgroup(mTilesPerSG, nTilesPerSG, elementBytes);

    // Check VGPR budget: all subgroups in a workgroup share the VGPR file.
    // Actually, each wave has its own VGPRs.  The constraint is per-wave.
    if (vgprsPerSG <= params_.maxVgprsPerSimd) {
      return sg;
    }
  }

  return 1;
}

//===----------------------------------------------------------------------===//
// 7.  Convert block sizes → GPUMMAHeuristicSeeds
//===----------------------------------------------------------------------===//
// The seeds that `deduceMMASchedule` expects are:
//
//   bestSubgroupCountPerWorkgroup  – number of subgroups (waves) per WG
//   bestMNTileCountPerSubgroup     – MMA-intrinsic tiles per SG in M*N
//   bestKTileCountPerSubgroup      – MMA-intrinsic tiles per SG in K
//   bestKElementCountPerSubgroup   – total K-elements per SG (= kTiles * mmaK)
//
// We derive them from the analytically computed block sizes.

GPUMMAHeuristicSeeds
GPUAnalyticalModel::blockSizesToSeeds(int64_t blockM, int64_t blockN,
                                      int64_t blockK, int64_t subgroups,
                                      int64_t pipelineDepth) const {
  int64_t mmaM = std::max(params_.mmaM, static_cast<int64_t>(1));
  int64_t mmaN = std::max(params_.mmaN, static_cast<int64_t>(1));
  (void)params_.mmaK; // mmaK used only via blockK

  // MN tiles across the entire workgroup.
  int64_t mTilesTotal = std::max(blockM / mmaM, static_cast<int64_t>(1));
  int64_t nTilesTotal = std::max(blockN / mmaN, static_cast<int64_t>(1));
  int64_t totalMNTiles = mTilesTotal * nTilesTotal;

  // Distribute across subgroups.
  int64_t mnTilesPerSubgroup =
      std::max(totalMNTiles / subgroups, static_cast<int64_t>(1));

  // K tiles per subgroup (pipeline depth).
  int64_t kTilesPerSubgroup = pipelineDepth;

  // K elements (= blockK, which is already a multiple of mmaK).
  int64_t kElements = blockK;

  GPUMMAHeuristicSeeds seeds;
  seeds.bestSubgroupCountPerWorkgroup = subgroups;
  seeds.bestMNTileCountPerSubgroup = mnTilesPerSubgroup;
  seeds.bestKTileCountPerSubgroup = kTilesPerSubgroup;
  seeds.bestKElementCountPerSubgroup = kElements;
  return seeds;
}

//===----------------------------------------------------------------------===//
// 8.  Top-level entry point
//===----------------------------------------------------------------------===//

std::optional<GPUMMAHeuristicSeeds>
GPUAnalyticalModel::computeOptimalSeeds(const GPUMatmulShapeType &problem,
                                        bool isGemm, bool scaled) {
  int64_t problemM = llvm::product_of(problem.mSizes);
  int64_t problemN = llvm::product_of(problem.nSizes);
  int64_t problemK = llvm::product_of(problem.kSizes);
  int64_t elementBitwidth = problem.aType.getIntOrFloatBitWidth();
  int64_t elementBytes = (elementBitwidth + 7) / 8;

  LDBG() << "=== tritonBLAS-style analytical model ===";
  LDBG() << "Problem: M=" << problemM << " N=" << problemN << " K=" << problemK
         << " elementBits=" << elementBitwidth;
  LDBG() << "Hardware: LDS=" << params_.sharedMemorySizeBytes
         << "B, cacheLine=" << params_.cacheLineSizeBytes
         << "B, MMA=" << params_.mmaM << "x" << params_.mmaN << "x"
         << params_.mmaK << ", subgroupSize=" << params_.subgroupSize
         << ", maxVGPR=" << params_.maxVgprsPerSimd
         << ", maxWaves=" << params_.maxWavesPerSimd;

  // Sanity: bail out if we have no useful hardware info.
  if (params_.sharedMemorySizeBytes <= 0 && params_.mmaM <= 0) {
    LDBG() << "Insufficient hardware info, returning nullopt";
    return std::nullopt;
  }

  // --- Step 1: Derive BLOCK_K from cache-line width ---
  int64_t blockK = deriveBlockK(elementBytes);
  LDBG() << "Step 1  BLOCK_K = " << blockK
         << "  (cacheLine=" << params_.cacheLineSizeBytes
         << " / elemBytes=" << elementBytes << ")";

  // --- Step 2: Derive initial BLOCK_M, BLOCK_N from LDS capacity ---
  // Start with a target pipeline depth of 2 (double-buffering) to get the
  // maximum possible MN block, then adjust.
  int64_t initialDepth = 2;
  int64_t maxBlockMN = deriveMaxBlockMN(blockK, elementBytes, initialDepth);

  // Choose blockM and blockN proportional to the problem aspect ratio,
  // but rounded to intrinsic multiples.
  int64_t mmaM = std::max(params_.mmaM, static_cast<int64_t>(1));
  int64_t mmaN = std::max(params_.mmaN, static_cast<int64_t>(1));

  // Compute how many intrinsic tiles fit in the MN budget.
  // Budget: blockM * blockK + blockN * blockK ≤ LDS / (elemBytes * stages)
  // With blockM ≈ blockN ≈ maxBlockMN, solve for each independently.
  // Use aspect ratio to distribute: if M >> N, give more to M.
  double aspectRatio =
      (problemN > 0) ? static_cast<double>(problemM) / problemN : 1.0;
  // Clamp aspect ratio to avoid extreme skew.
  aspectRatio = std::max(0.25, std::min(aspectRatio, 4.0));

  // blockM + blockN ≈ 2 * maxBlockMN (total budget for both)
  // Distribute: blockM = maxBlockMN * sqrt(aspect), blockN = maxBlockMN /
  // sqrt(aspect)
  double sqrtAspect = std::sqrt(aspectRatio);
  int64_t blockM = static_cast<int64_t>(maxBlockMN * sqrtAspect);
  int64_t blockN = static_cast<int64_t>(maxBlockMN / sqrtAspect);

  // Round to intrinsic multiples.
  blockM = std::max((blockM / mmaM) * mmaM, mmaM);
  blockN = std::max((blockN / mmaN) * mmaN, mmaN);

  // Clamp to problem size — no point tiling larger than the problem.
  blockM =
      std::min(blockM, static_cast<int64_t>(llvm::alignTo(problemM, mmaM)));
  blockN =
      std::min(blockN, static_cast<int64_t>(llvm::alignTo(problemN, mmaN)));

  LDBG() << "Step 2  initial BLOCK_M=" << blockM << " BLOCK_N=" << blockN
         << "  (maxBlockMN=" << maxBlockMN << ", aspect=" << aspectRatio << ")";

  // --- Step 3: Derive pipeline depth for chosen block sizes ---
  int64_t pipelineDepth =
      deriveMaxPipelineDepth(blockM, blockN, blockK, elementBytes);
  LDBG() << "Step 3  pipeline depth = " << pipelineDepth;

  // --- Step 4: Verify LDS fits, reduce block if needed ---
  // Iteratively reduce block sizes until the working set fits in LDS.
  while (sharedMemPerStage(blockM, blockN, blockK, elementBytes) *
             pipelineDepth >
         params_.sharedMemorySizeBytes) {
    if (blockM > blockN && blockM > mmaM) {
      blockM -= mmaM;
    } else if (blockN > mmaN) {
      blockN -= mmaN;
    } else if (pipelineDepth > 2) {
      --pipelineDepth;
    } else {
      break;
    }
  }
  // Ensure minimums.
  blockM = std::max(blockM, mmaM);
  blockN = std::max(blockN, mmaN);

  LDBG() << "Step 4  adjusted BLOCK_M=" << blockM << " BLOCK_N=" << blockN
         << " depth=" << pipelineDepth << " LDS_used="
         << sharedMemPerStage(blockM, blockN, blockK, elementBytes) *
                pipelineDepth;

  // --- Step 5: Derive subgroup count from occupancy model ---
  int64_t subgroups =
      deriveSubgroupCount(blockM, blockN, blockK, elementBytes, pipelineDepth);
  LDBG() << "Step 5  subgroups = " << subgroups;

  // --- Step 6: Convert to seeds ---
  GPUMMAHeuristicSeeds seeds =
      blockSizesToSeeds(blockM, blockN, blockK, subgroups, pipelineDepth);

  // Ensure all seeds are at least 1 to avoid assertion failures downstream.
  seeds.bestSubgroupCountPerWorkgroup =
      std::max(seeds.bestSubgroupCountPerWorkgroup, static_cast<int64_t>(1));
  seeds.bestMNTileCountPerSubgroup =
      std::max(seeds.bestMNTileCountPerSubgroup, static_cast<int64_t>(1));
  seeds.bestKTileCountPerSubgroup =
      std::max(seeds.bestKTileCountPerSubgroup, static_cast<int64_t>(2));
  seeds.bestKElementCountPerSubgroup =
      std::max(seeds.bestKElementCountPerSubgroup, static_cast<int64_t>(1));

  LDBG() << "=== Analytical seeds: subgroups="
         << seeds.bestSubgroupCountPerWorkgroup
         << ", MN_tiles=" << seeds.bestMNTileCountPerSubgroup
         << ", K_tiles=" << seeds.bestKTileCountPerSubgroup
         << ", K_elems=" << seeds.bestKElementCountPerSubgroup << " ===";

  return seeds;
}

} // namespace mlir::iree_compiler
