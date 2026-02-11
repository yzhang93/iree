// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_COMMON_GPU_GPUANALYTICALMODEL_H_
#define IREE_COMPILER_CODEGEN_COMMON_GPU_GPUANALYTICALMODEL_H_

#include "iree/compiler/Codegen/Common/GPU/GPUHeuristics.h"
#include "mlir/IR/Types.h"

namespace mlir::iree_compiler {

/// Architectural parameters extracted from the GPU target description.
/// These are the *only* inputs to the analytical model — all decisions
/// are derived from these hardware quantities plus the problem shape.
struct GPUArchitecturalParams {
  // --- Memory hierarchy ---
  int64_t sharedMemorySizeBytes = 0; // LDS per workgroup (bytes)
  int64_t cacheLineSizeBytes = 128;  // Global memory cache line (bytes)

  // --- Memory bandwidth ---
  float globalMemoryBandwidthGBps = 0.0f; // GB/s

  // --- Compute throughput ---
  // Peak TFLOPS for each element bitwidth (e.g., 16 → FP16 peak TFLOPS).
  llvm::DenseMap<int64_t, float> peakTflopsPerBitwidth;

  // --- Register file ---
  int64_t maxVgprsPerSimd = 256; // VGPRs per SIMD unit
  int64_t numSimdsPerWgp = 4;    // SIMD units (CUs) per WGP
  int64_t maxWavesPerSimd = 10;  // Max concurrent waves per SIMD

  // --- Workgroup / subgroup ---
  int64_t subgroupSize = 64; // Threads per wave/subgroup
  int64_t wgpCount = 0;      // Number of WGPs (Workgroup Processors)

  // --- MMA intrinsic shape (from the target's first available intrinsic) ---
  int64_t mmaM = 16;
  int64_t mmaN = 16;
  int64_t mmaK = 16;

  GPUArchitecturalParams() = default;
};

/// tritonBLAS-style analytical model for GPU GEMM configuration.
///
/// Instead of using fixed heuristic thresholds, all parameters are derived
/// from hardware specifications and the problem shape using closed-form
/// equations.  The model follows the approach described in the tritonBLAS
/// paper (arXiv:2512.04226):
///
///   1. Derive BLOCK_K from cache-line width and element size so that every
///      global memory transaction is fully utilised.
///   2. Derive BLOCK_M/BLOCK_N from the shared-memory capacity so that the
///      working-set (A-tile + B-tile) fits in LDS for a given pipeline depth.
///   3. Derive the subgroup (warp) count from an occupancy model that balances
///      register pressure and shared-memory usage against hardware limits.
///   4. Derive the pipeline depth (K-tile count) by dividing the available
///      LDS budget by the per-stage shared-memory footprint.
///
/// The model produces `GPUMMAHeuristicSeeds` that can be consumed by the
/// existing `deduceMMASchedule` infrastructure.
class GPUAnalyticalModel {
public:
  GPUAnalyticalModel(const GPUArchitecturalParams &params) : params_(params) {}

  /// Compute optimal seeds derived purely from hardware parameters and the
  /// problem shape.  Returns std::nullopt only when hardware parameters are
  /// insufficient (all zeros).
  std::optional<GPUMMAHeuristicSeeds>
  computeOptimalSeeds(const GPUMatmulShapeType &problem, bool isGemm,
                      bool scaled);

private:
  // ---------- Derived quantities (no magic constants) ----------

  /// BLOCK_K: number of K-elements that fills one cache line.
  ///   = cacheLineBytes / elementBytes
  int64_t deriveBlockK(int64_t elementBytes) const;

  /// Per-stage shared-memory footprint in bytes for a given tile.
  ///   = (blockM * blockK + blockN * blockK) * elementBytes
  int64_t sharedMemPerStage(int64_t blockM, int64_t blockN, int64_t blockK,
                            int64_t elementBytes) const;

  /// Maximum pipeline depth (stages) that fit in shared memory.
  ///   = floor(sharedMemorySize / sharedMemPerStage)
  int64_t deriveMaxPipelineDepth(int64_t blockM, int64_t blockN, int64_t blockK,
                                 int64_t elementBytes) const;

  /// Maximum square-ish MN block size that fits in shared memory for a given
  /// pipeline depth and BLOCK_K.
  ///   Solve: (blockMN + blockMN) * blockK * elementBytes * stages <= LDS
  ///   =>     blockMN <= LDS / (2 * blockK * elementBytes * stages)
  int64_t deriveMaxBlockMN(int64_t blockK, int64_t elementBytes,
                           int64_t pipelineDepth) const;

  /// Estimate VGPRs consumed per subgroup for a given tile.
  ///   Each subgroup holds an accumulator tile of size
  ///   (mTile * mmaM) * (nTile * mmaN) elements, stored in registers.
  ///   Plus input fragments and index overhead.
  int64_t estimateVgprsPerSubgroup(int64_t mTilesPerSubgroup,
                                   int64_t nTilesPerSubgroup,
                                   int64_t elementBytes) const;

  /// Derive subgroup count from occupancy constraints.
  ///   Target: maximise waves in flight while keeping VGPRs and LDS in budget.
  int64_t deriveSubgroupCount(int64_t blockM, int64_t blockN, int64_t blockK,
                              int64_t elementBytes,
                              int64_t pipelineDepth) const;

  /// Convert block sizes to seed values that deduceMMASchedule expects.
  GPUMMAHeuristicSeeds blockSizesToSeeds(int64_t blockM, int64_t blockN,
                                         int64_t blockK, int64_t subgroups,
                                         int64_t pipelineDepth) const;

  GPUArchitecturalParams params_;
};

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_CODEGEN_COMMON_GPU_GPUANALYTICALMODEL_H_
