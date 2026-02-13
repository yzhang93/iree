# Roofline-Based MMA Seed Selection — Results & Findings

## Overview

This document summarizes the implementation and evaluation of a **roofline-based candidate evaluation model** for MMA heuristic seed selection in IREE's GPU code generation pipeline. The goal is to replace hardcoded, manually-tuned lookup tables with a hardware-agnostic analytical model that derives tile size decisions from queryable GPU hardware parameters — achieving cross-GPU portability with **zero tuning constants**.

## Implementation

### Design (inspired by tritonBLAS / origami)

The model enumerates **30 seed candidates** from:
- `subgroupCount` ∈ {2, 4, 8}
- `mnTileCount` ∈ {2, 4, 8, 16, 32}
- `kTileCount` ∈ {2, 4}

Each candidate is scored using a 6-step roofline cost model:

1. **Tile dimension estimation** — derives `wgTileM`, `wgTileN`, `wgTileK` from seeds + intrinsic shape using the same sqrt-based distribution as `getOptimalMMASchedule`.
2. **Shared memory constraint** — hard filter; rejects candidates exceeding `maxWorkgroupMemoryBytes`.
3. **Occupancy** — computes total workgroups including split-K factor: `numWGs = ⌈M/wgTileM⌉ × ⌈N/wgTileN⌉ × splitK`, then `numTimesteps = ⌈numWGs / wgpCount⌉`.
4. **Work utilization** — padding efficiency: `workUtil = (M × N) / (M_padded × N_padded)`.
5. **Per-WG roofline cost** — `iterCost = max(memTime, compTime)` where bandwidth and compute are shared across `wgpCount` processors.
6. **Total cost** — `iterCost × kIters × numTimesteps / workUtil`, where `kIters = ⌈K/splitK/wgTileK⌉`.

### Intrinsic-Aware Seed Selection via Callback

Seed selection is integrated into `deduceMMASchedule` via a **`SeedSelector` callback**. Instead of choosing seeds blindly before knowing which MMA intrinsic will be used, the roofline model is invoked **once per intrinsic** inside `deduceMMASchedule`'s intrinsic loop. This ensures seeds are always optimized for the exact intrinsic shape (e.g., 16×16×16 vs 32×32×8).

Architecture:
- **`GPUHeuristics.h`** — defines `SeedSelector` callback type (no hardware-specific dependencies).
- **`GPUHeuristics.cpp`** — when `seedSelector` is provided, calls it per-intrinsic instead of using fixed default seeds. `adjustSeedsForWgpCount` still runs on the result.
- **`ConfigUtils.cpp`** — builds a roofline lambda capturing `RooflineHardwareInfo` and passes it as the `seedSelector`.

### Hardware Inputs (all from `IREE::GPU::TargetAttr`)

| Parameter | Source | MI300X Value |
|-----------|--------|-------------|
| `peakFlopsPerSec` | `chip.perfTflops[bitwidth]` | 1307.4 TFLOP/s (BF16) |
| `memBandwidthBytesPerSec` | `chip.memoryBandwidthTbps` | 662.5 GB/s |
| `maxSharedMemoryBytes` | `wgp.maxWorkgroupMemoryBytes` | 65536 B |
| `wgpCount` | `chip.wgpCount` | 304 |
| `subgroupSize` | `target.preferredSubgroupSize` | 64 |
| `intrinsics` | from `deduceMMASchedule` intrinsic loop | 16×16×16, 32×32×8, etc. |

**Zero manually-tuned constants.** All decisions derive from the hardware specification, the intrinsic shape passed by `deduceMMASchedule`, and the `splitReductionTripCnt` already computed by the compiler.

### Code Changes

Changes span 3 files:
- **`GPUHeuristics.h`** — added `SeedSelector` type alias and optional `seedSelector` parameter to `deduceMMASchedule`.
- **`GPUHeuristics.cpp`** — when `seedSelector` is present, invokes it per-intrinsic inside the loop.
- **`ConfigUtils.cpp`** — `RooflineHardwareInfo` struct (no intrinsic list), `extractRooflineHardwareInfo()`, `evaluateSeedCandidate()`, `selectBestSeeds()`, and roofline callback construction in `getMmaScheduleFromProblemAndTarget()`.

Flag: `--iree-codegen-use-roofline-seeds` (defaults to `true`).

Fallback: if hardware info is incomplete (missing `TargetAttr`, no chip spec, etc.), the model falls back to the existing hardcoded seeds automatically.

---

## Benchmark Results

**Platform:** AMD MI300X, 320 GEMM shapes (BF16), benchmarked via `iree-turbine` driver.
**Valid comparisons:** 319 shapes.

### Overall Performance

| Metric | Value |
|--------|-------|
| **Geometric mean speedup** | **1.2853×** (+28.5%) |
| **Weighted speedup (by execution time)** | **3.0977×** |
| Total baseline execution time | 2,177,392 μs |
| Total roofline execution time | 702,896 μs |
| Time saved by improvements (>5%) | 1,493,652 μs |
| Time lost by regressions (<-5%) | 14,672 μs |
| **Net time saved** | **1,474,496 μs** |
| **Time saved / time lost ratio** | **101.8×** |

### Distribution

| Category | Count | Percentage |
|----------|-------|-----------|
| Improvements (>5%) | 195 | 61.1% |
| Regressions (<-5%) | 80 | 25.1% |
| Neutral (±5%) | 44 | 13.8% |

### Top 10 Improvements

| Shape (M×K×N) | Baseline (μs) | Roofline (μs) | Speedup |
|---------------|---------------|---------------|---------|
| 16384×150000×4096 | 638,568.5 | 48,358.2 | **13.20×** |
| 4096×150000×16384 | 633,742.8 | 49,349.0 | **12.84×** |
| 150000×1134×2048 | 19,970.1 | 2,163.8 | **9.23×** |
| 16640×3840×3840 | 8,969.8 | 1,067.0 | **8.41×** |
| 4352×3840×3840 (addmm) | 2,492.4 | 307.4 | **8.11×** |
| 21760×3840×3840 | 11,507.5 | 1,424.1 | **8.08×** |
| 150000×1024×3072 | 16,890.4 | 2,111.1 | **8.00×** |
| 11520×3840×3840 | 5,982.9 | 748.3 | **7.99×** |
| 6400×3840×3840 | 3,490.4 | 447.0 | **7.81×** |
| 150000×2268×4096 | 68,629.9 | 8,932.9 | **7.68×** |

### Top 10 Regressions

| Shape (M×K×N) | Baseline (μs) | Roofline (μs) | Slowdown |
|---------------|---------------|---------------|----------|
| 4112×2048×2048 (addmm) | 134.4 | 385.6 | **2.87×** |
| 512×7680×512 | 41.1 | 83.0 | **2.02×** |
| 512×7680×304 | 41.5 | 81.5 | **1.96×** |
| 20×21760×3840 | 120.1 | 222.4 | **1.85×** |
| 128×2119936×128 | 520.7 | 938.6 | **1.80×** |
| 128×3670016×128 | 934.8 | 1,587.9 | **1.70×** |
| 18928×512×128 (addmm) | 18.3 | 30.9 | **1.69×** |
| 7680×2048×512 | 62.9 | 105.5 | **1.68×** |
| 4096×150000×105 | 3,002.3 | 4,956.8 | **1.65×** |
| 512×32768×128 | 49.6 | 80.0 | **1.61×** |

### Evolution Across Model Iterations

| Model Version | Geo Mean | Improvements (>5%) | Regressions (<-5%) |
|---------------|----------|--------------------|--------------------|
| v3: Original roofline | 1.2478× | 167 | 98 |
| v4: + Split-K awareness | 1.2732× | 170 | 90 |
| v5: + Callback (per-intrinsic seeds) | **1.2853×** | **195** | **80** |

Key improvements across iterations:
- **Split-K awareness** (+0.025× geo mean): Threading `splitReductionTripCnt` through the model correctly accounts for K-dimension parallelism, eliminating the worst regressions in large-K shapes.
- **Per-intrinsic callback** (+0.012× geo mean): Moving seed selection into `deduceMMASchedule` via `SeedSelector` callback ensures seeds are optimized for the exact intrinsic shape. Increased improvements from 170 to 195 and reduced regressions from 90 to 80.

---

## Analysis of Remaining Limitations

### Skinny / very-skinny shapes (M or N ≤ 256)

For shapes where one dimension is much smaller than the tile size, the model's `workUtil` penalty correctly identifies padding waste but sometimes still selects tiles that are slightly too large for the narrow dimension. The per-WG roofline cost `∝ (wgTileM + wgTileN)` favors the smallest tiles, which is usually correct for skinny shapes but can conflict with the `numTimesteps` term for well-occupied problems.

### Small MN shapes (M×N < 1M)

These shapes have moderate dimensions but are still under-occupied. The model can't capture micro-architectural effects (instruction scheduling, MFMA pipeline depth) that determine the optimal tile for these intermediate cases.

### L2 cache effect

For under-occupied problems (numWGs < wgpCount), MI300X's 256 MB L2 cache makes actual DRAM traffic nearly independent of tiling. The roofline model's memory term becomes approximately constant across candidates, limiting its ability to differentiate. The remaining differentiation comes from `workUtil` and `(wgTileM + wgTileN)` which provide reasonable but imperfect guidance.

---

## Summary

| Aspect | Assessment |
|--------|-----------|
| **Architecture agnosticism** | ✅ Zero tuning constants; all parameters from `TargetAttr` |
| **Large workload perf** | ✅ Significant gains (up to 13.2×) |
| **Overall weighted perf** | ✅ 3.10× weighted speedup |
| **Geometric mean** | ✅ 1.285× (+28.5%) |
| **Split-K awareness** | ✅ Correctly models split-K parallelism |
| **Per-intrinsic seeds** | ✅ Callback selects optimal seeds per intrinsic |
| **Net time impact** | ✅ 101.8× more time saved than lost |
| **Regression count** | ⚠️ 80 shapes regress >5% (mostly skinny/small shapes) |
| **Skinny/small shapes** | ⚠️ Systematic regressions, low absolute impact |

## Potential Future Improvements

1. **Skinny dimension handling** — when M or N is much smaller than the minimum tile size, consider clamping the tile to avoid excessive padding. This could be done by pre-filtering candidates where `workUtil < threshold` (though the threshold would need derivation from hardware).

2. **Cross-GPU validation** — test on NVIDIA (WMMA), other AMD architectures (RDNA), and Intel GPUs to verify portability before broadening the default.

3. **Hybrid approach** — for the remaining small-shape regressions, explore falling back to the hardcoded seeds when the problem is very small (total FLOPs below some hardware-derived threshold).
