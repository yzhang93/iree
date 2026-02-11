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

/// Architectural parameters that characterize GPU hardware capabilities.
struct GPUArchitecturalParams {
  // Cache sizes (in bytes)
  int64_t l1CacheSizeBytes = 0;
  int64_t l2CacheSizeBytes = 0;
  int64_t sharedMemorySizeBytes = 0;

  // Memory bandwidths (in Tbps - Terabits per second)
  float globalMemoryBandwidthTbps = 0.0f;
  float sharedMemoryBandwidthTbps = 0.0f;

  // Peak compute throughput (TFLOPS per bitwidth)
  llvm::DenseMap<int64_t, float> peakTflopsPerBitwidth;

  // Register limits
  int64_t maxVgprsPerWorkgroup = 0;
  int64_t maxSgprsPerWorkgroup = 0;

  // Subgroup and workgroup parameters
  int64_t subgroupSize = 0;
  int64_t wgpCount = 0;
  int64_t maxWorkgroupsPerWgp = 0;

  // Matrix core dimensions (typical intrinsic shapes)
  int64_t typicalMmaM = 0;
  int64_t typicalMmaN = 0;
  int64_t typicalMmaK = 0;

  // Cache line size (in bits)
  int64_t cacheLineSizeBits = 128 * 8; // Default: 128 bytes

  GPUArchitecturalParams() = default;
};

/// Analytical performance model for computing optimal MMA heuristic seeds.
/// Inspired by tritonBLAS approach to model cache hierarchy, memory bandwidth,
/// compute throughput, and register pressure.
class GPUAnalyticalModel {
public:
  GPUAnalyticalModel(const GPUArchitecturalParams &params) : params_(params) {}

  /// Main entry point: compute optimal seeds for a given problem.
  /// Returns std::nullopt if the model cannot compute valid seeds.
  std::optional<GPUMMAHeuristicSeeds>
  computeOptimalSeeds(const GPUMatmulShapeType &problem, bool isGemm,
                      bool scaled);

private:
  /// Estimate cache hit rate based on tile sizes and problem dimensions.
  /// Returns a value between 0.0 (no hits) and 1.0 (all hits).
  float estimateCacheHitRate(int64_t tileM, int64_t tileN, int64_t tileK,
                             int64_t problemM, int64_t problemN,
                             int64_t problemK);

  /// Compute memory bandwidth utilization for given tile sizes.
  /// Returns utilization as a fraction (0.0 to 1.0+).
  float computeMemoryBandwidthUtilization(int64_t tileM, int64_t tileN,
                                          int64_t tileK,
                                          int64_t elementBitwidth,
                                          float cacheHitRate);

  /// Estimate workgroup occupancy based on register pressure and shared memory.
  /// Returns occupancy as a fraction (0.0 to 1.0).
  float estimateOccupancy(int64_t subgroupCount, int64_t tileM, int64_t tileN,
                          int64_t tileK, int64_t elementBitwidth);

  /// Compute register pressure (VGPR usage) for given tile sizes.
  /// Returns estimated VGPR count per workgroup.
  int64_t computeRegisterPressure(int64_t subgroupCount, int64_t tileM,
                                  int64_t tileN, int64_t tileK,
                                  int64_t elementBitwidth);

  /// Estimate data reuse factor for cache modeling.
  /// Higher values indicate more temporal/spatial locality.
  float estimateDataReuse(int64_t tileM, int64_t tileN, int64_t tileK,
                          int64_t problemM, int64_t problemN, int64_t problemK);

  /// Compute optimal subgroup count per workgroup.
  int64_t computeOptimalSubgroupCount(int64_t problemM, int64_t problemN,
                                      int64_t problemK, int64_t elementBitwidth,
                                      bool isGemm, bool scaled);

  /// Compute optimal M/N tile count per subgroup.
  int64_t computeOptimalMNTileCount(int64_t problemM, int64_t problemN,
                                    int64_t problemK, int64_t elementBitwidth,
                                    int64_t subgroupCount,
                                    float arithmeticIntensity, bool isGemm,
                                    bool scaled);

  /// Compute optimal K tile count per subgroup.
  int64_t computeOptimalKTileCount(int64_t problemK, int64_t elementBitwidth,
                                   int64_t subgroupCount, bool isGemm);

  /// Compute optimal K element count per subgroup.
  int64_t computeOptimalKElementCount(int64_t elementBitwidth);

  /// Validate computed seeds against hardware limits.
  bool validateSeeds(const GPUMMAHeuristicSeeds &seeds, int64_t elementBitwidth,
                     int64_t numRhs = 1);

  GPUArchitecturalParams params_;
};

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_CODEGEN_COMMON_GPU_GPUANALYTICALMODEL_H_
