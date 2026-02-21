# Roofline-Based MMA Seed Selection — Results & Findings

## Overview

This document summarizes the implementation and evaluation of a **roofline-based candidate evaluation model** for MMA heuristic seed selection in IREE's GPU code generation pipeline. The goal is to replace hardcoded, manually-tuned lookup tables with a hardware-agnostic analytical model that derives tile size decisions from queryable GPU hardware parameters.

## Implementation

### Design (inspired by tritonBLAS / origami)

The model enumerates seed candidates from:
- `subgroupCount` ∈ {2, 4, 8} (filtered to ≥ `simdsPerWgp`)
- `mnTileCount` ∈ {2, 4, 8, 16}
- `kTileCount` ∈ {8, 4, 2} (iterated in descending order)

Each candidate is scored using a cache-aware roofline cost model:

1. **Tile dimension estimation** — derives `wgTileM`, `wgTileN`, `wgTileK` from seeds + intrinsic shape using the same sqrt-based distribution as `getOptimalMMASchedule`.
2. **Shared memory constraint** — hard filter; rejects candidates exceeding `maxWorkgroupMemoryBytes`.
3. **Register pressure constraint** — hard filter; rejects candidates where estimated VGPRs (accumulator + operand) exceed `vgprsPerSimd`, preventing catastrophic register spilling.
4. **Occupancy estimation** — computes LDS-based and register-based occupancy, capped at `kCostOccupancyCap = 2` for cost estimation (beyond 2 concurrent WGs per WGP, the marginal latency-hiding benefit is small).
5. **Work utilization** — padding efficiency: `workUtil = (M × N) / (M_padded × N_padded)`.
6. **L2-cache-aware memory model** — in a tiled GEMM, A tiles are reused across N-columns of workgroups and B tiles across M-rows. Per-WG bytes are divided by a reuse factor (capped at `kMaxCacheReuse = 8`) to model L2 hit rates.
7. **Per-WG roofline cost** — `iterCost = max(memTime, compTime)` where bandwidth and compute are shared across concurrent WGs.
8. **Total cost** — `iterCost × kIters × numTimesteps / workUtil`.

**kTileCounts are iterated in descending order** so that when multiple configs produce the same cost (common due to the cache model), the largest kTile wins — fewer K-loop iterations means less barrier synchronisation overhead.

### Post-Search Fixups

- **Subgroup fixup**: when the best config uses more subgroups than SIMDs per WGP (causing multiple waves per SIMD), the model searches for an equivalent config (same WG tile dimensions) with fewer subgroups for better instruction-level parallelism.

### Intrinsic-Aware Seed Selection via Callback

Seed selection is integrated into `deduceMMASchedule` via a **`SeedSelector` callback**. Instead of choosing seeds blindly before knowing which MMA intrinsic will be used, the roofline model is invoked **once per intrinsic** inside `deduceMMASchedule`'s intrinsic loop. This ensures seeds are always optimized for the exact intrinsic shape (e.g., 16×16×32 vs 32×32×16).

Architecture:
- **`GPUHeuristics.h`** — defines `SeedSelector` callback type (no hardware-specific dependencies).
- **`GPUHeuristics.cpp`** — when `seedSelector` is provided, calls it per-intrinsic instead of using fixed default seeds. `adjustSeedsForWgpCount` still runs on the result.
- **`ConfigUtils.cpp`** — builds a roofline lambda capturing `RooflineHardwareInfo` and passes it as the `seedSelector`.

### Hardware Inputs (all from `IREE::GPU::TargetAttr`)

| Parameter | Source |
|-----------|--------|
| `peakFlopsPerSec` | `chip.perfTflops[bitwidth]` |
| `memBandwidthBytesPerSec` | `chip.memoryBandwidthTbps` |
| `maxSharedMemoryBytes` | `wgp.maxWorkgroupMemoryBytes` |
| `wgpCount` | `chip.wgpCount` |
| `subgroupSize` | `target.preferredSubgroupSize` |
| `simdsPerWgp` | `wgp.simdsPerWgp` |
| `vgprsPerSimd` | `wgp.vgprSpaceBits / 32` |
| `intrinsics` | from `deduceMMASchedule` intrinsic loop |

All decisions derive from the hardware specification, the intrinsic shape passed by `deduceMMASchedule`, and the `splitReductionTripCnt` already computed by the compiler.

### Code Changes

Changes span 3 files:
- **`GPUHeuristics.h`** — added `SeedSelector` type alias and optional `seedSelector` parameter to `deduceMMASchedule`.
- **`GPUHeuristics.cpp`** — when `seedSelector` is present, invokes it per-intrinsic inside the loop.
- **`ConfigUtils.cpp`** — `RooflineHardwareInfo` struct, `extractRooflineHardwareInfo()`, `evaluateSeedCandidate()`, `selectBestSeeds()`, and roofline callback construction in `getMmaScheduleFromProblemAndTarget()`.

Flag: `--iree-codegen-use-roofline-seeds` (defaults to `true`).

Fallback: if hardware info is incomplete (missing `TargetAttr`, no chip spec, etc.), the model falls back to the existing hardcoded seeds automatically.

---

## Analysis of Known Limitations

### L2 cache modelling

The L2 cache reuse model uses a simple cap (`kMaxCacheReuse = 8`) for both A-tile and B-tile reuse. This is a coarse approximation — actual cache hit rates depend on the L2 cache size, WG dispatch order, and concurrent access patterns. For GPUs with very large L2 caches (e.g., MI300X's 256 MB Infinity Cache), the real reuse factor may be much higher, while for GPUs with smaller caches it may be lower.

### Occupancy cap

The `kCostOccupancyCap = 2` prevents the model from over-valuing high occupancy, which was causing regressions on MI355X (160 KB LDS allows occupancy up to 5–6 for small tiles). However, this cap may be too conservative for workloads that genuinely benefit from high occupancy (e.g., very large K with many K-loop iterations).

### Skinny / very-skinny shapes (M or N ≤ 256)

For shapes where one dimension is much smaller than the tile size, the model's `workUtil` penalty correctly identifies padding waste but sometimes still selects tiles that are slightly too large for the narrow dimension.

### Micro-architectural effects

The model cannot capture instruction scheduling, MFMA pipeline depth, or wave scheduling efficiency, which can affect performance for small/medium shapes.

---

## Summary

| Aspect | Assessment |
|--------|-----------|
| **Architecture agnosticism** | ✅ All parameters from `TargetAttr` |
| **Split-K awareness** | ✅ Correctly models split-K parallelism |
| **Per-intrinsic seeds** | ✅ Callback selects optimal seeds per intrinsic |
| **Register spilling prevention** | ✅ Hard VGPR limit rejects infeasible configs |
| **L2 cache awareness** | ✅ Models A/B tile reuse across WG rows/columns |
| **Occupancy modelling** | ✅ Capped to avoid over-valuing high occupancy |
| **Skinny/small shapes** | ⚠️ Potential regressions, low absolute impact |

## Potential Future Improvements

1. **Skinny dimension handling** — when M or N is much smaller than the minimum tile size, consider clamping the tile to avoid excessive padding.

2. **Cross-GPU validation** — test on NVIDIA (WMMA), other AMD architectures (RDNA), and Intel GPUs to verify portability before broadening the default.

3. **Dynamic cache reuse cap** — derive `kMaxCacheReuse` from the actual L2 cache size (if available as a hardware attribute) instead of using a fixed constant.

4. **Adaptive occupancy cap** — derive `kCostOccupancyCap` from the memory latency characteristics of the target GPU rather than using a fixed value of 2.
