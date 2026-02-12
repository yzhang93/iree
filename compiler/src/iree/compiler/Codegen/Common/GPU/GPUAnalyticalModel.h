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

/// Analytical model for GPU GEMM configuration seed selection.
///
/// The model classifies a GEMM problem into one of three regimes
/// (memory-bound, balanced, compute-bound) based on arithmetic intensity
/// relative to the machine's compute-to-memory-bandwidth ratio (the
/// "machine balance point").  Each regime maps to a fixed set of
/// empirically-validated seed values:
///
///   Memory-bound  → sg=2  mn=2   kT=4  kE=2·cacheLine/elemBytes
///   Balanced      → sg=4  mn=8   kT=4  kE=2·cacheLine/elemBytes
///   Compute-bound → sg=4  mn=16  kT=2  kE=cacheLine/(2·elemBytes)
///
/// The regime boundaries and seed values are derived from hardware
/// specifications following principles similar to the tritonBLAS approach
/// (arXiv:2512.04226):
///
///   • Arithmetic intensity (AI) = 2·M·N·K / ((M·K + K·N + M·N)·elemBytes)
///   • Machine balance = peakTFLOPS·1000 / (memBW_GBps · 2 · elemBytes)
///   • Memory-bound:  AI < 0.2 · balance   (most time in loads)
///   • Compute-bound: AI > 20  · balance   (most time in FMAs)
///   • Balanced:      everything in between
///
/// The seeds are *aspirational*: `deduceMMASchedule`'s
/// `fitScheduleInSharedMemory` will refine the actual tile counts to fit
/// within LDS.  Starting with these seeds lets the fitting logic converge
/// to the best achievable schedule.
class GPUAnalyticalModel {
public:
  GPUAnalyticalModel(const GPUArchitecturalParams &params) : params_(params) {}

  /// Compute optimal seeds derived from hardware parameters and problem shape.
  /// Returns std::nullopt only when hardware parameters are insufficient.
  std::optional<GPUMMAHeuristicSeeds>
  computeOptimalSeeds(const GPUMatmulShapeType &problem, bool isGemm,
                      bool scaled);

private:
  GPUArchitecturalParams params_;
};

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_CODEGEN_COMMON_GPU_GPUANALYTICALMODEL_H_
