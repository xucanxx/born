//go:build windows

// Package webgpu implements the WebGPU backend for GPU-accelerated tensor operations.
package webgpu

import (
	"fmt"
	"runtime"

	"github.com/born-ml/born/internal/tensor"
	"github.com/gogpu/gputypes"
	wgpu "github.com/gogpu/wgpu"
)

// ─────────────────────────────────────────────────────────────────────────────
// Shared encoder accumulator (Part 2)
// ─────────────────────────────────────────────────────────────────────────────

// getOrCreateEncoderLocked returns the active CommandEncoder, creating one if
// necessary. MUST be called with pendingMu held.
func (b *Backend) getOrCreateEncoderLocked() *wgpu.CommandEncoder {
	if b.activeBatch.encoder == nil {
		var err error
		b.activeBatch.encoder, err = b.device.CreateCommandEncoder(nil)
		if err != nil {
			panic("webgpu: CreateCommandEncoder: " + err.Error())
		}
	}
	return b.activeBatch.encoder
}

// finishActiveBatchLocked finalizes the active encoder — appending all pending
// CopyBufferToBuffer entries, finishing the encoder into a CommandBuffer, and
// appending it to b.pending. Resets activeBatch to zero value.
//
// MUST be called with pendingMu held. Safe to call when activeBatch.encoder is
// nil (no-op).
func (b *Backend) finishActiveBatchLocked() {
	if b.activeBatch.encoder == nil {
		return
	}

	enc := b.activeBatch.encoder

	// Append all buffered copies AFTER all compute passes and before Finish().
	// wgpu requires CopyBufferToBuffer to be outside a compute pass scope.
	for _, cp := range b.activeBatch.copies {
		enc.CopyBufferToBuffer(cp.src, 0, cp.dst, 0, cp.size)
	}

	cmdBuffer, err := enc.Finish()
	if err != nil {
		// On Finish error, release everything to avoid leaks and panic loudly.
		for _, buf := range b.activeBatch.resultBufs {
			buf.Release()
		}
		for _, bg := range b.activeBatch.bindGroups {
			bg.Release()
		}
		b.activeBatch = encoderBatch{}
		panic("webgpu: finishActiveBatch: encoder Finish: " + err.Error())
	}

	b.pending = append(b.pending, pendingSubmission{
		cmdBuffer:  cmdBuffer,
		resultBufs: b.activeBatch.resultBufs,
		bindGroups: b.activeBatch.bindGroups,
	})

	b.activeBatch = encoderBatch{}
}

// addComputePassToEncoder encodes a single compute dispatch into the shared
// encoder accumulator. It replaces the per-op pattern of:
//
//	encoder := CreateCommandEncoder()
//	computePass := encoder.BeginComputePass()
//	... SetPipeline / SetBindGroup / Dispatch / End ...
//	finishAndQueueLazy(encoder, resultBuf, stagingBuf, ...)
//
// The staging buffer copy is deferred to finishActiveBatchLocked so that all
// copies in the batch are emitted after all compute passes (wgpu requirement).
//
// Returns a lazy RawTensor backed by stagingBuf, exactly as finishAndQueueLazy
// did. The caller must NOT defer-release stagingBuf — ownership transfers to
// the lazy tensor.
//
// res.buffers and res.bindGroups follow the same ownership rules as
// finishAndQueueLazy: they must NOT be defer-released by the caller.
//
func (b *Backend) addComputePassToEncoder(
	pipeline *wgpu.ComputePipeline,
	bg *wgpu.BindGroup,
	workgroupsX, workgroupsY, workgroupsZ uint32,
	resultBuf *wgpu.Buffer,
	stagingBuf *wgpu.Buffer,
	resultSize uint64,
	shape tensor.Shape,
	dtype tensor.DataType,
	res lazyResources,
) (*tensor.RawTensor, error) {
	b.pendingMu.Lock()

	enc := b.getOrCreateEncoderLocked()

	computePass, err := enc.BeginComputePass(nil)
	if err != nil {
		b.pendingMu.Unlock()
		// Release caller-owned resources on failure.
		resultBuf.Release()
		stagingBuf.Release()
		for _, buf := range res.buffers {
			buf.Release()
		}
		bg.Release()
		return nil, fmt.Errorf("addComputePassToEncoder: BeginComputePass: %w", err)
	}
	computePass.SetPipeline(pipeline)
	computePass.SetBindGroup(0, bg, nil)
	computePass.Dispatch(workgroupsX, workgroupsY, workgroupsZ)
	if err := computePass.End(); err != nil {
		b.pendingMu.Unlock()
		resultBuf.Release()
		stagingBuf.Release()
		for _, buf := range res.buffers {
			buf.Release()
		}
		bg.Release()
		panic(fmt.Sprintf("webgpu: addComputePassToEncoder: compute pass End: %v", err))
	}

	// Record the copy from resultBuf → stagingBuf to be emitted before Finish.
	b.activeBatch.copies = append(b.activeBatch.copies, bufferCopyEntry{
		src:  resultBuf,
		dst:  stagingBuf,
		size: resultSize,
	})

	// Track all resources for post-Submit release.
	// resultBuf is the intermediate storage buffer; it must stay alive until Submit.
	b.activeBatch.resultBufs = append(b.activeBatch.resultBufs, resultBuf)
	b.activeBatch.resultBufs = append(b.activeBatch.resultBufs, res.buffers...)
	b.activeBatch.bindGroups = append(b.activeBatch.bindGroups, bg)
	b.activeBatch.bindGroups = append(b.activeBatch.bindGroups, res.bindGroups...)
	b.activeBatch.count++
	count := b.activeBatch.count

	// Auto-flush at threshold to prevent Windows TDR timeout.
	// finishActiveBatchLocked moves the completed encoder to b.pending; then
	// flushCommands submits all pending in one queue.Submit.
	if count >= maxPendingBeforeFlush {
		b.finishActiveBatchLocked()
	}

	b.pendingMu.Unlock()

	// Auto-flush: if we just sealed the batch, submit it now.
	// (count >= threshold was handled inside the lock; flushCommands re-checks
	// and is a fast no-op if there is nothing to flush.)
	if count >= maxPendingBeforeFlush {
		b.flushCommands()
	}

	return b.createLazyResult(stagingBuf, resultSize, shape, dtype)
}

// ─────────────────────────────────────────────────────────────────────────────
// Input buffer cache (Part 1)
// ─────────────────────────────────────────────────────────────────────────────

// getOrCreateInputBuffer returns a GPU storage buffer for the given CPU tensor,
// creating and caching it on first access. Subsequent calls with the same
// *RawTensor pointer return the cached buffer without re-uploading.
//
// ONLY CPU tensors (t.GPUData() == nil) are eligible for caching. Lazy GPU
// tensors are transient intermediates — they always go through copyGPUBuffer
// with no caching, identical to the old createBufferFromTensor path.
//
// Cached buffers are NEVER added to lazyResources.buffers and are therefore
// never released by finishActiveBatchLocked. They are released in
// clearInputBufferCache(), which is called from Backend.Release().
//
// Thread-safe: uses a separate RWMutex (inputBufferCache.mu) so cache reads
// do not contend with pendingMu.
func (b *Backend) getOrCreateInputBuffer(t *tensor.RawTensor) *wgpu.Buffer {
	// Lazy (GPU-backed) tensors: cannot cache — they are transient results whose
	// staging buffer will be released after the lazy tensor is read. Always
	// perform the GPU→GPU copy path, same as before.
	if gpuData := t.GPUData(); gpuData != nil && !gpuData.IsRealized() {
		existingBuffer := (*wgpu.Buffer)(gpuData.BufferPtr())
		result := b.copyGPUBuffer(existingBuffer, gpuData.Size())
		runtime.KeepAlive(gpuData)
		return result
	}

	// CPU tensor fast path: check cache first.
	b.inputBufferCache.mu.RLock()
	if cb, ok := b.inputBufferCache.cache[t]; ok {
		b.inputBufferCache.mu.RUnlock()
		return cb.buffer // Cache hit — reuse existing GPU buffer.
	}
	b.inputBufferCache.mu.RUnlock()

	// Cache miss — upload CPU data to a new GPU storage buffer.
	buf := b.createBuffer(t.Data(), gputypes.BufferUsageStorage|gputypes.BufferUsageCopySrc)
	size := uint64(t.ByteSize()) //nolint:gosec // G115: ByteSize is non-negative

	b.inputBufferCache.mu.Lock()
	if b.inputBufferCache.cache == nil {
		b.inputBufferCache.cache = make(map[*tensor.RawTensor]*cachedBuffer)
	}
	// Double-check: another goroutine may have inserted while we were uploading.
	// If so, release our newly created buffer and return the existing one.
	if existing, ok := b.inputBufferCache.cache[t]; ok {
		b.inputBufferCache.mu.Unlock()
		buf.Release()
		return existing.buffer
	}
	b.inputBufferCache.cache[t] = &cachedBuffer{buffer: buf, size: size}
	b.inputBufferCache.mu.Unlock()

	return buf
}

// clearInputBufferCache releases all cached GPU buffers and clears the cache.
// Called automatically from Backend.Release(). May also be called explicitly
// between training steps when weight tensors change (e.g. after optimizer.Step
// replaces tensors with new objects).
func (b *Backend) clearInputBufferCache() {
	b.inputBufferCache.mu.Lock()
	defer b.inputBufferCache.mu.Unlock()

	for _, cb := range b.inputBufferCache.cache {
		cb.buffer.Release()
	}
	b.inputBufferCache.cache = nil
}

// ClearInputBufferCache is the exported version of clearInputBufferCache.
// Call this between training steps if weight tensors are replaced by new
// *RawTensor objects (e.g. after in-place optimizer updates that produce new
// tensors). Forgetting to call this after weight replacement causes stale
// GPU data to be used in forward passes.
func (b *Backend) ClearInputBufferCache() {
	b.clearInputBufferCache()
}

// inputBufferCacheSize returns the current number of entries in the input
// buffer cache. Intended for testing and observability only.
func (b *Backend) inputBufferCacheSize() int {
	b.inputBufferCache.mu.RLock()
	defer b.inputBufferCache.mu.RUnlock()
	return len(b.inputBufferCache.cache)
}

// activeBatchCount returns the number of compute passes currently accumulated
// in the active (not-yet-flushed) encoder batch. Intended for testing only.
func (b *Backend) activeBatchCount() int {
	b.pendingMu.Lock()
	defer b.pendingMu.Unlock()
	return b.activeBatch.count
}

