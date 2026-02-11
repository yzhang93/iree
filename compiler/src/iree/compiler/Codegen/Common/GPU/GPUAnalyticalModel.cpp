// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/GPU/GPUAnalyticalModel.h"

#include "llvm/Support/DebugLog.h"
#include "llvm/Support/MathExtras.h"

#define DEBUG_TYPE "iree-gpu-analytical-model"

using namespace mlir::iree_compiler;

namespace mlir::iree_compiler {

//===----------------------------------------------------------------------===//
// Cache Hierarchy Modeling
//===----------------------------------------------------------------------===//

float GPUAnalyticalModel::estimateDataReuse(int64_t tileM, int64_t tileN,
                                            int64_t tileK, int64_t problemM,
                                            int64_t problemN,
                                            int64_t problemK) {
  // Estimate reuse based on how many times each element is accessed
  // For matmul: A is reused N times, B is reused M times
  float reuseA = static_cast<float>(tileN) / static_cast<float>(problemN);
  float reuseB = static_cast<float>(tileM) / static_cast<float>(problemM);
  return (reuseA + reuseB) / 2.0f;
}

float GPUAnalyticalModel::estimateCacheHitRate(int64_t tileM, int64_t tileN,
                                               int64_t tileK, int64_t problemM,
                                               int64_t problemN,
                                               int64_t problemK) {
  // Estimate cache hit rate based on tile size relative to cache capacity
  int64_t tileSizeBytes = (tileM * tileK + tileN * tileK) *
                          8; // Assume 8 bytes per element for estimation
  float dataReuse =
      estimateDataReuse(tileM, tileN, tileK, problemM, problemN, problemK);

  // If tile fits in L1, expect high hit rate
  float l1FitRatio = params_.l1CacheSizeBytes > 0
                         ? static_cast<float>(tileSizeBytes) /
                               static_cast<float>(params_.l1CacheSizeBytes)
                         : 1.0f;
  float l2FitRatio = params_.l2CacheSizeBytes > 0
                         ? static_cast<float>(tileSizeBytes) /
                               static_cast<float>(params_.l2CacheSizeBytes)
                         : 1.0f;

  // Base hit rate from cache fit
  float baseHitRate = 0.0f;
  if (l1FitRatio < 0.5f) {
    baseHitRate = 0.9f; // Fits comfortably in L1
  } else if (l1FitRatio < 1.0f) {
    baseHitRate = 0.7f; // Fits in L1 but tight
  } else if (l2FitRatio < 0.5f) {
    baseHitRate = 0.5f; // Fits in L2
  } else if (l2FitRatio < 1.0f) {
    baseHitRate = 0.3f; // Fits in L2 but tight
  } else {
    baseHitRate = 0.1f; // Doesn't fit well in cache
  }

  // Boost hit rate based on data reuse
  float reuseBoost = std::min(0.3f, dataReuse * 0.1f);
  return std::min(1.0f, baseHitRate + reuseBoost);
}

//===----------------------------------------------------------------------===//
// Memory Bandwidth Modeling
//===----------------------------------------------------------------------===//

float GPUAnalyticalModel::computeMemoryBandwidthUtilization(
    int64_t tileM, int64_t tileN, int64_t tileK, int64_t elementBitwidth,
    float cacheHitRate) {
  if (params_.globalMemoryBandwidthTbps <= 0.0f) {
    return 0.0f;
  }

  // Estimate bytes transferred per FMA operation
  // Each FMA requires: 1 element from A, 1 element from B
  int64_t bytesPerFMA = (elementBitwidth / 8) * 2; // A and B
  int64_t totalFMAs = tileM * tileN * tileK;

  // Account for cache hits (reduce effective memory traffic)
  float effectiveBytes =
      static_cast<float>(totalFMAs * bytesPerFMA) * (1.0f - cacheHitRate);

  // Convert to Tbps (assuming 1 cycle per FMA for bandwidth calculation)
  // This is a simplified model - actual calculation would need cycle counts
  float bandwidthRequiredTbps = effectiveBytes * 8.0f / 1e12f; // Simplified

  return bandwidthRequiredTbps / params_.globalMemoryBandwidthTbps;
}

//===----------------------------------------------------------------------===//
// Register Pressure Analysis
//===----------------------------------------------------------------------===//

int64_t GPUAnalyticalModel::computeRegisterPressure(int64_t subgroupCount,
                                                    int64_t tileM,
                                                    int64_t tileN,
                                                    int64_t tileK,
                                                    int64_t elementBitwidth) {
  // Estimate VGPR usage based on tile sizes
  // This is a simplified model - actual register allocation is more complex
  int64_t elementsPerSubgroup = (tileM * tileN) / subgroupCount;
  int64_t bytesPerSubgroup = elementsPerSubgroup * (elementBitwidth / 8);

  // Rough estimate: each element in registers uses some VGPRs
  // Assume 4 bytes per VGPR (typical for many GPUs)
  int64_t estimatedVgprs = bytesPerSubgroup / 4;

  // Add overhead for accumulators, indices, etc.
  estimatedVgprs += 64; // Base overhead

  return estimatedVgprs * subgroupCount;
}

float GPUAnalyticalModel::estimateOccupancy(int64_t subgroupCount,
                                            int64_t tileM, int64_t tileN,
                                            int64_t tileK,
                                            int64_t elementBitwidth) {
  if (params_.maxVgprsPerWorkgroup == 0 || params_.maxWorkgroupsPerWgp == 0) {
    return 0.5f; // Default occupancy if limits unknown
  }

  int64_t vgprUsage = computeRegisterPressure(subgroupCount, tileM, tileN,
                                              tileK, elementBitwidth);
  float vgprOccupancy = static_cast<float>(params_.maxVgprsPerWorkgroup) /
                        static_cast<float>(vgprUsage);

  // Also consider shared memory occupancy
  int64_t sharedMemUsage =
      (tileM * tileK + tileN * tileK) * (elementBitwidth / 8);
  float sharedMemOccupancy =
      params_.sharedMemorySizeBytes > 0
          ? static_cast<float>(params_.sharedMemorySizeBytes) /
                static_cast<float>(sharedMemUsage)
          : 1.0f;

  // Occupancy is limited by the most constrained resource
  float occupancy = std::min(vgprOccupancy, sharedMemOccupancy);
  occupancy =
      std::min(occupancy, static_cast<float>(params_.maxWorkgroupsPerWgp));

  return std::max(0.0f, std::min(1.0f, occupancy));
}

//===----------------------------------------------------------------------===//
// Core Analytical Formulas
//===----------------------------------------------------------------------===//

int64_t GPUAnalyticalModel::computeOptimalSubgroupCount(
    int64_t problemM, int64_t problemN, int64_t problemK,
    int64_t elementBitwidth, bool isGemm, bool scaled) {
  // Base calculation: consider problem size and arithmetic intensity
  // Larger problems can benefit from more parallelism
  int64_t problemSize = problemM * problemN;
  float problemSizeFactor = 1.0f;

  // Scale based on problem size
  if (problemSize > 1000000) { // > 1M elements
    problemSizeFactor = 1.5f;
  } else if (problemSize > 100000) { // > 100K elements
    problemSizeFactor = 1.2f;
  } else if (problemSize < 10000) {  // < 10K elements
    problemSizeFactor = 0.7f;        // Smaller problems need fewer subgroups
  } else if (problemSize < 100000) { // 10K-100K elements
    problemSizeFactor = 0.9f;        // Medium-small problems: slightly reduce
  }

  // Target occupancy: Choose based on problem characteristics
  // Rationale for 50-75% range:
  // - Too low (<50%): Underutilizes GPU, poor latency hiding
  // - Too high (>75%): Resource contention, register spilling, shared memory
  // pressure
  // - 50-75% is a sweet spot that balances parallelism and resource efficiency
  // - 62.5% (0.625) is the middle point, providing a balanced default
  //
  // The value 0.625 was chosen as:
  // 1. Middle of the 50-75% range mentioned in the plan
  // 2. A reasonable default that works across different problem sizes
  // 3. Can be tuned based on performance data (future work)
  float targetOccupancy = 0.625f; // Default: middle of 50-75% range

  // Adjust based on problem characteristics (can be refined with performance
  // data) For now, use a simple heuristic: larger problems can use higher
  // occupancy because they have better arithmetic intensity and can hide
  // latency better
  if (problemSize > 1000000) {
    targetOccupancy = 0.70f; // 70% for very large problems (compute-bound)
  } else if (problemSize < 10000) {
    targetOccupancy = 0.55f; // 55% for small problems (memory-bound, need more
                             // resources per workgroup)
  }

  int64_t baseSubgroups =
      params_.maxWorkgroupsPerWgp > 0
          ? static_cast<int64_t>(params_.maxWorkgroupsPerWgp * targetOccupancy)
          : 4; // Default

  // Apply problem size factor
  int64_t maxSubgroups =
      static_cast<int64_t>(baseSubgroups * problemSizeFactor);

  // Consider WGP count for workgroup distribution
  if (params_.wgpCount > 0) {
    // Prefer subgroup counts that distribute well across WGPs
    // But don't reduce too much for large problems
    int64_t wgpLimit = params_.wgpCount * 2;
    if (problemSize > 100000) {
      // For large problems, allow more subgroups even if it exceeds wgpCount *
      // 2
      wgpLimit = std::max(wgpLimit, static_cast<int64_t>(params_.wgpCount * 3));
    }
    maxSubgroups = std::min(maxSubgroups, wgpLimit);
  }

  // For scaled matmuls, prefer more subgroups for parallelism
  if (scaled) {
    maxSubgroups = std::max(maxSubgroups, static_cast<int64_t>(8));
    // For large scaled matmuls, prefer even more
    if (problemSize > 1000000) {
      maxSubgroups = std::max(maxSubgroups, static_cast<int64_t>(12));
    }
  }

  // For convolutions, favor more subgroups for latency hiding
  if (!isGemm) {
    maxSubgroups = std::max(maxSubgroups, static_cast<int64_t>(8));
  }

  // Clamp to reasonable bounds
  return std::max(static_cast<int64_t>(2),
                  std::min(maxSubgroups, static_cast<int64_t>(16)));
}

int64_t GPUAnalyticalModel::computeOptimalMNTileCount(
    int64_t problemM, int64_t problemN, int64_t problemK,
    int64_t elementBitwidth, int64_t subgroupCount, float arithmeticIntensity,
    bool isGemm, bool scaled) {
  // Higher for compute-bound (large AI), lower for memory-bound
  // Arithmetic intensity = ops / bytes
  float baseTileCount = 4.0f;

  // Scale based on arithmetic intensity
  if (arithmeticIntensity > 100.0f) {
    baseTileCount = 32.0f; // Very compute-bound
  } else if (arithmeticIntensity > 10.0f) {
    baseTileCount = 16.0f; // Compute-bound
  } else if (arithmeticIntensity > 1.0f) {
    baseTileCount = 8.0f; // Balanced
  } else {
    baseTileCount = 4.0f; // Memory-bound
  }

  // For scaled matmuls, prefer larger tiles
  if (scaled) {
    baseTileCount = std::max(baseTileCount, 32.0f);
  }

  // Constrain by shared memory capacity
  int64_t maxTileCount = 64; // Reasonable upper bound
  if (params_.sharedMemorySizeBytes > 0) {
    // Estimate max tiles that fit in shared memory
    // Rough estimate: each tile element needs storage
    int64_t bytesPerTileElement = elementBitwidth / 8;
    int64_t estimatedMaxTiles = params_.sharedMemorySizeBytes /
                                (bytesPerTileElement * 1024); // Rough estimate
    maxTileCount = std::min(maxTileCount, estimatedMaxTiles);
  }

  int64_t tileCount = static_cast<int64_t>(baseTileCount);
  return std::max(static_cast<int64_t>(2), std::min(tileCount, maxTileCount));
}

int64_t GPUAnalyticalModel::computeOptimalKTileCount(int64_t problemK,
                                                     int64_t elementBitwidth,
                                                     int64_t subgroupCount,
                                                     bool isGemm) {
  // Balance reduction loop overhead vs. memory access efficiency
  // For compute-bound: fewer K tiles (less overhead)
  // For memory-bound: more K tiles (better memory access patterns)

  int64_t baseKTiles = 4;

  // For high K dimensions, prefer fewer K tiles to reduce reduction overhead
  // This helps with regressions in high-K problems (e.g., k=512)
  if (problemK > 256) {
    baseKTiles = 2; // Reduce from 4 to 2 for high K
  } else if (problemK > 128) {
    baseKTiles = 3; // Moderate reduction for medium-high K
  } else if (problemK < 64) {
    // For very small K, use minimal tiles to avoid overhead
    baseKTiles = 2;
  }

  // For convolutions, prefer more K tiles for latency hiding
  if (!isGemm) {
    baseKTiles = 4;
  }

  // Align with cache line sizes for optimal memory access
  // Prefer multiples that align well with cache lines
  int64_t cacheLineElements = params_.cacheLineSizeBits / elementBitwidth;
  if (cacheLineElements > 0) {
    // Round to nearest multiple of cache line elements
    baseKTiles = llvm::alignTo(baseKTiles, cacheLineElements);
  }

  // Ensure K tiles don't exceed problem K dimension
  if (problemK > 0 && baseKTiles > problemK / 4) {
    baseKTiles = std::max(static_cast<int64_t>(2), problemK / 4);
  }

  return std::max(static_cast<int64_t>(2),
                  std::min(baseKTiles, static_cast<int64_t>(8)));
}

int64_t
GPUAnalyticalModel::computeOptimalKElementCount(int64_t elementBitwidth) {
  // Align to cache line boundaries
  int64_t cacheLineElements = params_.cacheLineSizeBits / elementBitwidth;
  if (cacheLineElements <= 0) {
    cacheLineElements = 16; // Default fallback
  }

  // Use a multiple of cache line size for prefetching and memory coalescing
  return cacheLineElements * 2; // 2x cache line for better prefetching
}

bool GPUAnalyticalModel::validateSeeds(const GPUMMAHeuristicSeeds &seeds,
                                       int64_t elementBitwidth,
                                       int64_t numRhs) {
  // Validate shared memory usage
  // Note: Seeds are heuristics (counts/parameters), not actual tile sizes.
  // The actual shared memory calculation happens in deduceMMASchedule() which
  // uses the real tile sizes from the computed schedule. Therefore, we use
  // a very lenient check here - the real validation happens later.
  // We only reject obviously invalid seeds (e.g., extremely large values).
  if (params_.sharedMemorySizeBytes > 0) {
    // Seeds are parameters that will be used to compute actual tile sizes.
    // A very rough upper bound: assume worst-case scenario where each seed
    // parameter directly translates to tile elements.
    // This is intentionally lenient - actual validation is in
    // deduceMMASchedule.
    int64_t bytesPerElement = (elementBitwidth + 7) / 8;
    // Very rough estimate - seeds are not actual tile sizes
    int64_t estimatedSharedMem =
        seeds.bestSubgroupCountPerWorkgroup * seeds.bestMNTileCountPerSubgroup *
        seeds.bestKElementCountPerSubgroup * bytesPerElement * 2; // A and B
    // Only reject if estimate is clearly way too large (10x the limit)
    // This allows seeds to pass through - deduceMMASchedule will do real
    // validation
    int64_t threshold = params_.sharedMemorySizeBytes * 10;
    if (estimatedSharedMem > threshold) {
      LDBG() << "Analytical model: seeds clearly exceed shared memory limit ("
             << estimatedSharedMem << " > " << threshold
             << ", limit=" << params_.sharedMemorySizeBytes << ")";
      return false;
    }
  }

  // Validate register limits (very lenient - seeds are heuristics, not final
  // sizes)
  if (params_.maxVgprsPerWorkgroup > 0) {
    // Estimate actual tile sizes from counts for validation
    int64_t typicalM = params_.typicalMmaM > 0 ? params_.typicalMmaM : 16;
    int64_t typicalN = params_.typicalMmaN > 0 ? params_.typicalMmaN : 16;
    int64_t estimatedTileM = seeds.bestMNTileCountPerSubgroup * typicalM;
    int64_t estimatedTileN = seeds.bestMNTileCountPerSubgroup * typicalN;
    int64_t estimatedTileK = seeds.bestKElementCountPerSubgroup;

    int64_t estimatedVgprs = computeRegisterPressure(
        seeds.bestSubgroupCountPerWorkgroup, estimatedTileM, estimatedTileN,
        estimatedTileK, elementBitwidth);

    // Use very lenient threshold (10x) since this is a rough estimate
    if (estimatedVgprs > params_.maxVgprsPerWorkgroup * 10) {
      LDBG() << "Analytical model: seeds clearly exceed VGPR limit ("
             << estimatedVgprs << " > " << (params_.maxVgprsPerWorkgroup * 10)
             << ", limit=" << params_.maxVgprsPerWorkgroup << ")";
      return false;
    }
  }

  // Basic sanity checks
  if (seeds.bestSubgroupCountPerWorkgroup <= 0 ||
      seeds.bestMNTileCountPerSubgroup <= 0 ||
      seeds.bestKTileCountPerSubgroup <= 0 ||
      seeds.bestKElementCountPerSubgroup <= 0) {
    LDBG() << "Analytical model: seeds have invalid values";
    return false;
  }

  return true;
}

std::optional<GPUMMAHeuristicSeeds>
GPUAnalyticalModel::computeOptimalSeeds(const GPUMatmulShapeType &problem,
                                        bool isGemm, bool scaled) {
  LDBG() << "Analytical model: computing seeds for problem M="
         << llvm::product_of(problem.mSizes)
         << " N=" << llvm::product_of(problem.nSizes)
         << " K=" << llvm::product_of(problem.kSizes)
         << " gemmSize=" << problem.gemmSize;

  int64_t problemM = llvm::product_of(problem.mSizes);
  int64_t problemN = llvm::product_of(problem.nSizes);
  int64_t problemK = llvm::product_of(problem.kSizes);
  int64_t elementBitwidth = problem.aType.getIntOrFloatBitWidth();

  // Validate problem dimensions
  if (problemM <= 0) {
    problemM = 1;
  }
  if (problemN <= 0) {
    problemN = 1;
  }
  if (problemK <= 0) {
    problemK = 1;
  }
  if (elementBitwidth <= 0) {
    elementBitwidth = 16;
  }

  // Detect problem characteristics for special handling
  int64_t problemSize = problemM * problemN;
  bool isSmallProblem = (problemSize < 100000) || (problemK < 256);
  bool isBatched = (problem.mSizes.size() > 1) || (problem.nSizes.size() > 1) ||
                   (problem.kSizes.size() > 1) ||
                   (problem.batchSizes.size() > 0);

  LDBG() << "Analytical model: problem characteristics - size=" << problemSize
         << ", K=" << problemK << ", isSmall=" << isSmallProblem
         << ", isBatched=" << isBatched;

  // Use analytical model formulas directly instead of hardcoded gemmSize
  // replication This allows the model to adapt to problem characteristics
  // automatically

  // Compute arithmetic intensity for tile sizing decisions
  // Arithmetic intensity = ops / bytes = (M * N * K) / (K * (M + N) *
  // bytes_per_element) Simplified: (M * N) / ((M + N) * bytes_per_element)
  int64_t bytesPerElement = (elementBitwidth + 7) / 8; // Round up
  float arithmeticIntensity = 0.0f;
  if (problemM > 0 && problemN > 0 && bytesPerElement > 0) {
    int64_t totalOps = problemM * problemN * problemK;
    int64_t totalBytes =
        (problemM * problemK + problemN * problemK) * bytesPerElement;
    if (totalBytes > 0) {
      arithmeticIntensity =
          static_cast<float>(totalOps) / static_cast<float>(totalBytes);
    }
  }
  // Fallback for edge cases
  if (arithmeticIntensity <= 0.0f) {
    arithmeticIntensity = 10.0f; // Default to compute-bound assumption
  }

  // Use analytical formulas to compute optimal seeds
  int64_t subgroupCount = computeOptimalSubgroupCount(
      problemM, problemN, problemK, elementBitwidth, isGemm, scaled);

  int64_t mnTileCount = computeOptimalMNTileCount(
      problemM, problemN, problemK, elementBitwidth, subgroupCount,
      arithmeticIntensity, isGemm, scaled);

  int64_t kTileCount = computeOptimalKTileCount(problemK, elementBitwidth,
                                                subgroupCount, isGemm);

  int64_t kElementCount = computeOptimalKElementCount(elementBitwidth);

  // Apply special handling for small problems
  if (isSmallProblem) {
    LDBG() << "Analytical model: applying small problem optimizations";
    // For small problems, use more conservative seeds
    // Reduce subgroup count to avoid overhead
    subgroupCount = std::min(subgroupCount, static_cast<int64_t>(4));
    // Reduce tile sizes to better match problem size
    mnTileCount = std::min(mnTileCount, static_cast<int64_t>(8));
    // For very small K, reduce K tiles
    if (problemK < 128) {
      kTileCount = std::min(kTileCount, static_cast<int64_t>(2));
    }
  }

  // Apply special handling for batched matmuls
  if (isBatched) {
    LDBG() << "Analytical model: applying batched matmul optimizations";
    // For batched ops, prefer fewer subgroups to reduce coordination overhead
    subgroupCount = std::min(subgroupCount, static_cast<int64_t>(4));
    // But prefer larger tiles to amortize overhead
    mnTileCount = std::max(mnTileCount, static_cast<int64_t>(8));
  }

  // Add lower bounds: ensure seeds don't exceed problem size constraints
  // Subgroup count should be reasonable (already clamped to 2-16)
  // MN tile count should not exceed problem dimensions
  int64_t maxMNTiles = std::max(problemM, problemN) / 16; // Rough estimate
  if (maxMNTiles > 0) {
    mnTileCount = std::min(mnTileCount, maxMNTiles);
  }
  // K tile count should not exceed K dimension
  if (problemK > 0) {
    kTileCount = std::min(kTileCount, problemK / 8); // Rough estimate
  }
  // Ensure minimum values
  mnTileCount = std::max(mnTileCount, static_cast<int64_t>(2));
  kTileCount = std::max(kTileCount, static_cast<int64_t>(2));

  // Ensure all computed values are valid
  if (subgroupCount <= 0) {
    subgroupCount = 2;
  }
  if (mnTileCount <= 0) {
    mnTileCount = 2;
  }
  if (kTileCount <= 0) {
    kTileCount = 2;
  }
  if (kElementCount <= 0) {
    kElementCount = 8;
  }

  GPUMMAHeuristicSeeds seeds;
  seeds.bestSubgroupCountPerWorkgroup = subgroupCount;
  seeds.bestMNTileCountPerSubgroup = mnTileCount;
  seeds.bestKTileCountPerSubgroup = kTileCount;
  seeds.bestKElementCountPerSubgroup = kElementCount;

  // Validation is lenient - let deduceMMASchedule do the real validation
  // Always return seeds to avoid breaking compilation
  if (!validateSeeds(seeds, elementBitwidth, problem.numHorizontallyFusedOps)) {
    LDBG() << "Analytical model: validation failed but returning seeds anyway "
              "(lenient validation)";
    // Still return seeds - let deduceMMASchedule do the real validation
  }

  LDBG() << "Analytical model: computed seeds - subgroups=" << subgroupCount
         << ", MN tiles=" << mnTileCount << ", K tiles=" << kTileCount
         << ", K elements=" << kElementCount;

  return seeds;
}

} // namespace mlir::iree_compiler
