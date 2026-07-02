# Security Policy

## Supported Versions

Born ML Framework is currently in initial release. We provide security updates for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 0.1.0   | :white_check_mark: |
| < 0.1.0 | :x:                |

Future releases will follow semantic versioning with security backports for major versions.

## Reporting a Vulnerability

We take security seriously. If you discover a security vulnerability in Born ML Framework, please report it responsibly.

### How to Report

**DO NOT** open a public GitHub issue for security vulnerabilities.

Instead, please report security issues by:

1. **Private Security Advisory** (preferred):
   https://github.com/xucanxx/born/security/advisories/new

2. **Direct contact** to maintainers:
   Create a private GitHub issue or contact via discussions

### What to Include

Please include the following information in your report:

- **Description** of the vulnerability
- **Steps to reproduce** the issue (include minimal code example)
- **Affected versions** (which versions are impacted)
- **Potential impact** (DoS, memory corruption, gradient manipulation, etc.)
- **Suggested fix** (if you have one)
- **Your contact information** (for follow-up questions)

### Response Timeline

- **Initial Response**: Within 48-72 hours
- **Triage & Assessment**: Within 1 week
- **Fix & Disclosure**: Coordinated with reporter

We aim to:
1. Acknowledge receipt within 72 hours
2. Provide an initial assessment within 1 week
3. Work with you on a coordinated disclosure timeline
4. Credit you in the security advisory (unless you prefer to remain anonymous)

## Security Considerations for ML Framework

Born ML Framework processes numerical data and executes tensor operations. This introduces specific security risks.

### 1. Memory Safety in Tensor Operations

**Risk**: Unsafe memory access in tensor operations can lead to crashes or memory corruption.

**Attack Vectors**:
- Integer overflow in shape calculations or strides
- Buffer overflow when reading/writing tensor data
- Out-of-bounds access in indexing operations
- Memory exhaustion via massive tensor allocations

**Mitigation in Library**:
- ✅ Bounds checking on all tensor operations
- ✅ Shape validation before memory allocation
- ✅ Stride calculation with overflow detection
- ✅ Safe indexing with panic recovery
- 🔄 Fuzzing and property-based testing (planned for v0.2.0)

**User Recommendations**:
```go
// ❌ BAD - Don't trust user-provided shapes without validation
tensor := born.Zeros(userProvidedShape, backend)

// ✅ GOOD - Validate shapes and sizes first
if !isValidShape(userShape) || totalSize(userShape) > maxAllowedSize {
    return errors.New("invalid tensor shape")
}
tensor := born.Zeros(validatedShape, backend)
```

### 2. Adversarial Model Inputs

**Risk**: Malicious inputs can exploit model weaknesses or cause unexpected behavior.

**Attack Vectors**:
- Adversarial examples causing misclassification
- Input shape mismatch causing panics
- NaN/Inf injection causing gradient explosions
- Extremely large values causing overflow

**Mitigation**:
- ✅ Input validation in NN modules
- ✅ Shape compatibility checks
- ✅ NaN/Inf detection in gradient computation
- 🔄 Input sanitization utilities (planned)

**User Best Practices**:
```go
// ✅ Validate input shapes
if !input.Shape().Equal(model.ExpectedInputShape()) {
    return errors.New("invalid input shape")
}

// ✅ Check for NaN/Inf
if containsNaN(input) || containsInf(input) {
    return errors.New("invalid input values")
}

// ✅ Clip input ranges
input = clip(input, -10.0, 10.0)
```

### 3. Gradient Computation Safety

**Risk**: Gradient computation can encounter numerical instability or trigger vulnerabilities.

**Attack Vectors**:
- Gradient explosion (unbounded gradients)
- Gradient vanishing (underflow to zero)
- NaN propagation in backward pass
- Circular gradient graphs causing infinite loops

**Mitigation**:
- ✅ Gradient clipping support in optimizers
- ✅ NaN detection in backward pass
- ✅ Cycle detection in computation graph
- ✅ Numerical stability in loss functions (e.g., LogSumExp for CrossEntropy)

**Current Limits**:
- Max gradient magnitude: User-configurable clipping
- Max computation graph depth: 10,000 operations
- Tape size limit: Configurable via backend

### 4. Resource Exhaustion

**Risk**: ML operations can consume excessive memory or CPU resources.

**Attack Vectors**:
- Massive tensor allocations (memory exhaustion)
- Deeply nested computation graphs (stack overflow)
- Infinite training loops (CPU exhaustion)
- Large batch sizes causing OOM

**Mitigation**:
- Memory allocation limits enforced by OS
- Computation graph depth limits
- User-controlled batch sizes
- Gradient checkpointing for large models (planned)

**User Best Practices**:
```go
// ✅ Validate tensor sizes
totalSize := 1
for _, dim := range shape {
    totalSize *= dim
}
if totalSize > maxAllowedElements {
    return errors.New("tensor too large")
}

// ✅ Limit batch sizes
if batchSize > maxBatchSize {
    return errors.New("batch size too large")
}

// ✅ Use gradient accumulation for large batches
for i := 0; i < numMicroBatches; i++ {
    loss := model.Forward(microBatch[i])
    grads := backend.Backward(loss)
    accumulateGradients(grads)
}
optimizer.StepWithAccumulated(accumulatedGrads)
```

### 5. Integer Overflow in Shape Calculations

**Risk**: Large dimensions can cause integer overflow when computing strides or total sizes.

**Example Attack**:
```
Shape: [1000000, 1000000, 1000]
Total size: 1e15 elements (overflows int64)
Result: Small buffer allocated, large data read → crash
```

**Mitigation**:
- All shape calculations checked for overflow
- Safe multiplication with overflow detection
- Maximum dimension size: 2^31-1 per dimension
- Maximum total elements: 2^50 (~1 petabyte)

**Current Limits**:
- Max dimension value: 2^31-1
- Max total tensor elements: 2^50
- Max tensor size: Limited by available memory

### 6. Model Poisoning (Training Data Attacks)

**Risk**: Malicious training data can poison model weights.

**User Responsibility**:
- Validate training data sources
- Implement data augmentation carefully
- Monitor training metrics for anomalies
- Use differential privacy if needed

**Framework Support**:
- ✅ Gradient clipping to limit poisoning impact
- 🔄 Differential privacy utilities (planned for v0.2.0)
- 🔄 Robust loss functions (planned)

### 7. Zero External Dependencies

**Security Advantage**: Born ML Framework has zero external dependencies in the core library.

**Benefits**:
- ✅ No supply chain attacks via dependencies
- ✅ Complete control over code security
- ✅ No hidden vulnerabilities from third-party code
- ✅ Pure Go implementation (memory-safe language)

**Testing Dependencies** (dev only):
- No runtime dependencies
- Standard library only

## Security Best Practices for Users

### Input Validation

Always validate untrusted inputs:

```go
// Validate input shapes
func validateInput(input *tensor.Tensor, expectedShape tensor.Shape) error {
    if !input.Shape().Equal(expectedShape) {
        return fmt.Errorf("invalid shape: got %v, expected %v",
            input.Shape(), expectedShape)
    }
    return nil
}

// Validate numerical stability
func validateTensor(t *tensor.Tensor) error {
    data := t.Raw().AsFloat32()
    for _, v := range data {
        if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
            return errors.New("tensor contains NaN or Inf")
        }
    }
    return nil
}
```

### Resource Limits

Set limits for untrusted operations:

```go
// Limit tensor sizes
const maxTensorSize = 1e9 // 1 billion elements

if tensorSize(shape) > maxTensorSize {
    return errors.New("tensor too large")
}

// Limit batch sizes
const maxBatchSize = 1024

if batchSize > maxBatchSize {
    return errors.New("batch size too large")
}
```

### Error Handling

Always check errors - failures may indicate attacks:

```go
// ❌ BAD - Ignoring errors
model := born.NewLinear(784, 10, backend)
output := model.Forward(input)

// ✅ GOOD - Proper error handling
model := born.NewLinear(784, 10, backend)
if err := validateInput(input, model.ExpectedShape()); err != nil {
    return fmt.Errorf("input validation failed: %w", err)
}

output := model.Forward(input)
if err := validateTensor(output); err != nil {
    return fmt.Errorf("output validation failed: %w", err)
}
```

## Known Security Considerations

### 1. Numerical Stability

**Status**: Active mitigation via stable algorithms.

**Risk Level**: Medium

**Description**: Floating-point operations can lose precision or produce NaN/Inf values.

**Mitigation**:
- Numerically stable loss functions (LogSumExp for CrossEntropy)
- Gradient clipping support
- NaN/Inf detection in critical paths

### 2. Memory Safety

**Status**: Inherent from Go's memory safety.

**Risk Level**: Low

**Description**: Go provides memory safety, but unsafe operations exist.

**Mitigation**:
- No `unsafe` package usage in core tensor operations
- Bounds checking on all array accesses
- Panic recovery for critical operations

### 3. Concurrency Safety

**Status**: Race detector enabled in CI.

**Risk Level**: Low

**Description**: Concurrent access to shared tensors can cause data races.

**Mitigation**:
- Thread-safe backend operations
- Race detector in CI pipeline
- Documentation of thread-safety guarantees

## Security Testing

### Current Testing

- ✅ Unit tests with edge cases (NaN, Inf, overflow)
- ✅ Integration tests with realistic models
- ✅ Race detector in CI (-race flag)
- ✅ golangci-lint with 34+ security-focused linters
- ✅ Bounds checking in all tensor operations

### Planned for v0.2.0

- 🔄 Fuzzing with go-fuzz
- 🔄 Property-based testing with gopter
- 🔄 Static analysis with gosec
- 🔄 SAST scanning in CI
- 🔄 Adversarial robustness testing

## Security Disclosure History

No security vulnerabilities have been reported or fixed yet (project is in initial release v0.1.0).

When vulnerabilities are addressed, they will be listed here with:
- **CVE ID** (if assigned)
- **Affected versions**
- **Fixed in version**
- **Severity** (Critical/High/Medium/Low)
- **Credit** to reporter

## Security Contact

- **GitHub Security Advisory**: https://github.com/xucanxx/born/security/advisories/new
- **Public Issues** (for non-sensitive bugs): https://github.com/xucanxx/born/issues
- **Discussions**: https://github.com/xucanxx/born/discussions

## Bug Bounty Program

Born ML Framework does not currently have a bug bounty program. We rely on responsible disclosure from the security community.

If you report a valid security vulnerability:
- ✅ Public credit in security advisory (if desired)
- ✅ Acknowledgment in CHANGELOG
- ✅ Recognition in README contributors section
- ✅ Priority review and quick fix

---

**Thank you for helping keep Born ML Framework secure!** 🔒

*Security is a continuous process. We improve our security posture with each release.*
