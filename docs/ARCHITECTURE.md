# Born ML Framework - Architecture

**Status**: Living Document
**Last Updated**: 2026-05-17

---

## Overview

Born is a layered ML framework following Burn (Rust) architecture with Go adaptations.

```
┌─────────────────────────────────────────────────────┐
│  Application: models/llama, examples/mnist, HRM     │
├─────────────────────────────────────────────────────┤
│  generate: TextGenerator, Sampling, KV-Cache        │
├─────────────────────────────────────────────────────┤
│  nn: Linear, Attention, GQA, SwiGLU, RMSNorm, RoPE │
├─────────────────────────────────────────────────────┤
│  autodiff: Gradient tape, backward ops              │
├─────────────────────────────────────────────────────┤
│  tensor: Tensor[T, B], RawTensor, Shape, Backend    │
├──────────────────┬──────────────────────────────────┤
│  backend/cpu     │  backend/webgpu (Vulkan)         │
└──────────────────┴──────────────────────────────────┘
```

---

## Backend Interface

All tensor operations route through the `Backend` interface (`internal/tensor/backend.go`). CPU and WebGPU backends implement the same interface — switching backends changes *where* computation runs without changing *what* is computed.

```go
type Backend interface {
    // Element-wise: Add, Sub, Mul, Div, MulScalar, ...
    // Matrix: MatMul, BatchMatMul
    // Shape: Reshape, Transpose, Cat, Chunk, Expand
    // Reduction: Sum, SumDim, MeanDim, Argmax
    // Activation: Softmax, SiLU, Sigmoid, ReLU, Tanh
    // Indexing: Gather, Embedding, SelectAdd, ScatterAdd
    // Comparison: Greater, Equal, Where
    Name() string
    Device() Device
}
```

---

## Autodiff (Automatic Differentiation)

### Decorator Pattern

```go
base := cpu.New()                   // or webgpu.New()
backend := autodiff.New(base)       // Wraps any backend with gradient tracking
```

`AutodiffBackend` wraps any `Backend` and records operations on a gradient tape during forward pass. Backward traverses the tape in reverse, computing gradients via chain rule.

### Backward = Forward Ops Composition

All backward operations are implemented by composing forward backend ops. Tensors never leave the compute device during backward:

```
Forward:  x → backend.MatMul → backend.SiLU → backend.Add → loss
Backward: ← backend.Mul ← backend.Sigmoid ← backend.MatMul ← grad
```

This follows the Burn (Rust) architecture where `B::float_matmul()`, `B::float_mul()`, etc. are used in both forward and backward passes.

**No CPU fallback**: Prior to v0.8.2, some backward ops (SiLU, CrossEntropy, Embedding, etc.) read tensor data to CPU via `AsFloat32()` and computed gradients in Go loops. This forced GPU→CPU synchronization and broke the pipeline. v0.8.2 migrated all 7 affected ops to forward composition per [ADR-009](dev/ADR-009-backward-ops-composition.md).

### Gradient Tape

```go
backend.Tape().StartRecording()     // Begin tracking
output := model.Forward(input)      // Operations recorded
grads := autodiff.Backward(output)  // Reverse-mode differentiation
backend.Tape().StopRecording()
```

The tape stops recording during backward to prevent double-recording of gradient computation ops.

---

## Tensor System

### Typed Tensors

```go
type Tensor[T DType, B Backend] struct {
    raw     *RawTensor    // Low-level data
    backend B             // Compute backend
}
```

Generic type parameters provide compile-time safety:
- `T` constrains element type (`float32`, `float64`, `int32`, etc.)
- `B` constrains backend (enables backend-specific optimizations)

### Lazy GPU Evaluation

WebGPU backend uses lazy evaluation — operations queue GPU dispatches without reading results back to CPU. Data transfer only happens when `Data()` or `AsFloat32()` is explicitly called.

```
GPU Op → GPU Op → GPU Op → AsFloat32()
  ↓         ↓         ↓         ↓
[lazy]   [lazy]   [lazy]   [readback]
```

---

## CPU Backend Performance

### Parallel BatchMatMul

Batches are independent — parallelized via `sync.WaitGroup` + goroutines. Threshold: B ≤ 4 sequential, B > 4 parallel. Workers: `runtime.NumCPU()`.

### Cache-Tiled Blocked MatMul

i-block→k-block→j-block loop order keeps both A and B blocks in L1 cache. Block size: 64 (float32, 16KB), 32 (float64, 8KB). 3-5x speedup for matrices > 64×64.

### AVX2 SIMD (Go 1.26+, experimental)

Optional SIMD micro-kernel via `goexperiment.simd` + `simd/archsimd`. 4-row × 16-wide register block with FMA. Build with `GOEXPERIMENT=simd go build`. Scalar fallback compiles without the flag. 3.5x speedup on AVX2 hardware.

---

## WebGPU Backend

Pure Go GPU backend via [gogpu/wgpu](https://github.com/gogpu/wgpu) — zero CGO, zero runtime dependencies.

- **Compute shaders**: Hand-written WGSL (not compiler DSL like Burn's CubeCL)
- **Batched dispatch**: Lazy ops queue command buffers; single `queue.Submit` on first `Data()` access. Reduces 50+ submits per forward pass to 1.
- **Buffer pool**: Reuses GPU memory allocations
- **Pipeline cache**: Caches compiled compute pipelines
- **Vulkan primary**: `BackendsVulkan` for compute workloads

---

## Model Loading

```
GGUF file → gguf.ParseFile → TensorConverter → models/llama.LoadGGUF
                                    ↓
                              Dequantize (Q4_K, Q8_0, F16, ...)
                                    ↓
                              RawTensor → Model weights
```

`models/llama` provides end-to-end LLaMA inference with injectable attention for research experiments.

---

## Key Design Decisions

| Decision | Rationale | ADR |
|---|---|---|
| Backend interface in `internal/tensor` | Avoids import cycle | — |
| Decorator pattern for autodiff | Composable, testable | ADR-004 |
| Hand-written WGSL shaders | Control over GPU kernels | ADR-004 |
| Core API over HAL-direct for wgpu | Stability, portability | ADR-005 |
| Backward via forward composition | GPU-native gradients, Burn alignment | ADR-009 |
| CPU parallel + GPU batching + SIMD | Performance parity with references | ADR-010 |

Full ADR list: `docs/dev/ADR-*.md`

---

## References

- [Burn Framework](https://github.com/tracel-ai/burn) — Rust ML, primary architecture reference
- [gogpu/wgpu](https://github.com/gogpu/wgpu) — Pure Go WebGPU backend
- [ROADMAP.md](../ROADMAP.md) — Version strategy and milestones
- [PHILOSOPHY.md](PHILOSOPHY.md) — Design principles
