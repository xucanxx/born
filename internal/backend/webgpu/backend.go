//go:build windows

// Package webgpu implements the WebGPU backend for GPU-accelerated tensor operations.
// Uses gogpu/wgpu (github.com/gogpu/wgpu) for pure Go, zero-CGO WebGPU bindings.
package webgpu

import (
	"context"
	"fmt"
	"os"
	"sync"
	"time"
	"unsafe"

	"github.com/born-ml/born/internal/tensor"
	"github.com/gogpu/gputypes"
	wgpu "github.com/gogpu/wgpu"
	_ "github.com/gogpu/wgpu/hal/allbackends"
)

// pipelineEntry caches a compute pipeline together with its layouts.
// The pipeline layout must remain alive because Vulkan references it
// during vkCmdBindDescriptorSets on every SetBindGroup call.
type pipelineEntry struct {
	pipeline       *wgpu.ComputePipeline
	layout         *wgpu.BindGroupLayout
	pipelineLayout *wgpu.PipelineLayout
}

// pendingSubmission holds a finished command buffer that has not yet been
// submitted to the GPU queue, plus the intermediate result buffers that must
// remain alive until after queue.Submit returns (BUG-LAZY-DEFER-RELEASE).
//
// Each lazy op produces one pendingSubmission via finishAndQueueLazy. All
// pending submissions are flushed in a single queue.Submit call when any
// tensor's Data() triggers ReadGPUBuffer.
type pendingSubmission struct {
	cmdBuffer  *wgpu.CommandBuffer
	resultBufs []*wgpu.Buffer    // released after queue.Submit completes
	bindGroups []*wgpu.BindGroup // released after queue.Submit completes
}

// Backend implements tensor operations on GPU using WebGPU.
type Backend struct {
	instance *wgpu.Instance
	adapter  *wgpu.Adapter
	device   *wgpu.Device
	queue    *wgpu.Queue

	// Shader and pipeline cache
	shaders   map[string]*wgpu.ShaderModule
	pipelines map[string]pipelineEntry
	mu        sync.RWMutex

	// Device info
	adapterInfo *wgpu.AdapterInfo

	// Buffer pool for memory management
	bufferPool *BufferPool

	// Lazy mode: when true, operations return lazy tensors that keep data on GPU
	// until Data() is explicitly called. This is the key optimization for
	// Phase 3 Integration - eliminates readBuffer() bottleneck.
	// Default: true for optimal performance.
	LazyMode bool

	// Pending command buffers queued for batched submission.
	// Populated by finishAndQueueLazy; drained by flushCommands.
	// Protected by pendingMu. This is the core of the batched-dispatch
	// optimization: instead of 1 Submit per op (~500 µs each), all pending
	// command buffers are submitted in a single queue.Submit call when the
	// first Data() access triggers ReadGPUBuffer.
	pending   []pendingSubmission
	pendingMu sync.Mutex

	// Memory tracking
	memoryStats struct {
		totalAllocatedBytes uint64
		peakMemoryBytes     uint64
		activeBuffers       int64
		mu                  sync.RWMutex
	}
}

// New creates a new WebGPU backend.
// Returns an error if WebGPU is not available or initialization fails.
func New() (*Backend, error) {
	// Create WebGPU instance. Vulkan is the primary compute backend —
	// stable across all platforms and GPU vendors.
	instance, err := wgpu.CreateInstance(&wgpu.InstanceDescriptor{
		Backends: wgpu.BackendsVulkan,
	})
	if err != nil {
		return nil, fmt.Errorf("webgpu: failed to create instance: %w", err)
	}

	// Request adapter (GPU).
	adapter, err := instance.RequestAdapter(&wgpu.RequestAdapterOptions{
		PowerPreference: gputypes.PowerPreferenceHighPerformance,
	})
	if err != nil {
		instance.Release()
		return nil, fmt.Errorf("webgpu: failed to request adapter: %w", err)
	}

	// Get adapter info. In gogpu/wgpu, Info() returns AdapterInfo by value.
	info := adapter.Info()

	// Request device.
	device, err := adapter.RequestDevice(nil)
	if err != nil {
		adapter.Release()
		instance.Release()
		return nil, fmt.Errorf("webgpu: failed to request device: %w", err)
	}

	// Get default queue. In gogpu/wgpu the queue is accessed via device.Queue().
	queue := device.Queue()
	if queue == nil {
		device.Release()
		adapter.Release()
		instance.Release()
		return nil, fmt.Errorf("webgpu: failed to get queue")
	}

	b := &Backend{
		instance:    instance,
		adapter:     adapter,
		device:      device,
		queue:       queue,
		shaders:     make(map[string]*wgpu.ShaderModule),
		pipelines:   make(map[string]pipelineEntry),
		adapterInfo: &info,
		bufferPool:  NewBufferPool(device),
		LazyMode:    true, // Default: lazy mode enabled for optimal performance
	}

	return b, nil
}

// SetLazyMode enables or disables lazy evaluation mode.
// When enabled (default), operations return lazy tensors that keep data on GPU
// until Data() is explicitly called. This dramatically improves performance
// by eliminating unnecessary GPU→CPU transfers.
// When disabled, operations immediately transfer results to CPU (slower).
func (b *Backend) SetLazyMode(enabled bool) {
	b.LazyMode = enabled
}

// flushCommands submits all pending command buffers to the GPU in a single
// queue.Submit call, then releases the intermediate result buffers that were
// kept alive for the duration of the submission.
//
// Called by ReadGPUBuffer before Map — ensures the GPU has received all
// commands that produce data for the staging buffer being mapped.
//
// If there are no pending submissions, this is a fast no-op (single mutex
// acquire + nil check).
func (b *Backend) flushCommands() {
	b.pendingMu.Lock()
	if len(b.pending) == 0 {
		b.pendingMu.Unlock()
		return
	}
	pending := b.pending
	b.pending = nil
	b.pendingMu.Unlock()

	// Collect all command buffers for a single Submit call.
	// queue.Submit is variadic: Submit(cmds ...*CommandBuffer).
	cmdBufs := make([]*wgpu.CommandBuffer, len(pending))
	for i, p := range pending {
		cmdBufs[i] = p.cmdBuffer
	}

	if _, err := b.queue.Submit(cmdBufs...); err != nil {
		panic("webgpu: flushCommands: submit failed: " + err.Error())
	}

	// Release intermediate resources now that Submit has registered them
	// with the GPU's destroy queue (lastSubmissionIndex updated). wgpu defers
	// the actual HAL destruction until the GPU completes this submission index.
	for _, p := range pending {
		for _, buf := range p.resultBufs {
			buf.Release()
		}
		for _, bg := range p.bindGroups {
			bg.Release()
		}
	}
}

// Release releases all WebGPU resources.
// Must be called when the backend is no longer needed.
func (b *Backend) Release() {
	// Ensure GPU is fully idle before destroying resources.
	// Without this, rapid create/destroy cycles (e.g. test suites) can
	// overwhelm the driver on iGPUs with shared memory.
	if b.device != nil {
		b.device.Poll(wgpu.PollWait)
	}

	b.mu.Lock()
	defer b.mu.Unlock()

	// Release buffer pool
	if b.bufferPool != nil {
		b.bufferPool.Clear()
		b.bufferPool = nil
	}

	// Release pipelines and their associated layouts.
	for _, entry := range b.pipelines {
		entry.pipeline.Release()
		entry.layout.Release()
		if entry.pipelineLayout != nil {
			entry.pipelineLayout.Release()
		}
	}
	b.pipelines = nil

	// Release shaders
	for _, s := range b.shaders {
		s.Release()
	}
	b.shaders = nil

	// Release WebGPU objects.
	// Note: Queue is owned by Device in gogpu/wgpu and released via device.Release().
	b.queue = nil
	if b.device != nil {
		b.device.Release()
		b.device = nil
	}
	if b.adapter != nil {
		b.adapter.Release()
		b.adapter = nil
	}
	if b.instance != nil {
		b.instance.Release()
		b.instance = nil
	}
}

// Name returns the backend name.
func (b *Backend) Name() string {
	if b.adapterInfo != nil {
		return fmt.Sprintf("WebGPU (%s)", b.adapterInfo.Name)
	}
	return "WebGPU"
}

// Device returns the compute device.
func (b *Backend) Device() tensor.Device {
	return tensor.WebGPU
}

// AdapterInfo returns information about the GPU adapter.
func (b *Backend) AdapterInfo() *wgpu.AdapterInfo {
	return b.adapterInfo
}

// IsAvailable checks if WebGPU with compute shader support is available.
// Returns false on software renderers that don't support compute pipelines,
// and also returns false if the underlying driver panics (e.g., missing GPU
// on CI runners). The panic recovery prevents process crashes on headless
// systems such as GitHub Actions Windows runners that have no Vulkan driver.
func IsAvailable() (available bool) {
	// Recover from panics that originate inside the wgpu DLL on machines
	// with no GPU or no Vulkan driver installed. Without this guard,
	// CreateInstance/RequestAdapter can raise an access violation that
	// propagates as a Go panic and crashes the entire test binary before
	// any t.Skip() can execute.
	defer func() {
		if r := recover(); r != nil {
			available = false
		}
	}()

	backend, err := New()
	if err != nil {
		return false
	}
	defer backend.Release()

	// Verify compute shaders actually work by creating a minimal pipeline.
	// Software renderers pass adapter/device creation but fail here.
	shader, err := backend.device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label: "availability-check",
		WGSL:  "@compute @workgroup_size(1) fn main() {}",
	})
	if err != nil {
		return false
	}
	defer shader.Release()

	bgl, err := backend.device.CreateBindGroupLayout(&wgpu.BindGroupLayoutDescriptor{})
	if err != nil {
		return false
	}
	defer bgl.Release()

	pl, err := backend.device.CreatePipelineLayout(&wgpu.PipelineLayoutDescriptor{
		BindGroupLayouts: []*wgpu.BindGroupLayout{bgl},
	})
	if err != nil {
		return false
	}
	defer pl.Release()

	pipeline, err := backend.device.CreateComputePipeline(&wgpu.ComputePipelineDescriptor{
		Label: "availability-check", Layout: pl, Module: shader, EntryPoint: "main",
	})
	if err != nil {
		return false
	}
	pipeline.Release()

	return true
}

// ListAdapters returns information about all available GPU adapters.
func ListAdapters() ([]*wgpu.AdapterInfo, error) {
	instance, err := wgpu.CreateInstance(nil)
	if err != nil {
		return nil, fmt.Errorf("webgpu: failed to create instance: %w", err)
	}
	defer instance.Release()

	// WebGPU spec doesn't expose adapter enumeration; return the default adapter.
	adapter, err := instance.RequestAdapter(nil)
	if err != nil {
		return nil, fmt.Errorf("webgpu: no adapters available: %w", err)
	}
	defer adapter.Release()

	// In gogpu/wgpu, Info() returns AdapterInfo by value (no error).
	info := adapter.Info()
	return []*wgpu.AdapterInfo{&info}, nil
}

// MemoryStats represents GPU memory usage statistics.
type MemoryStats struct {
	// Total bytes allocated since backend creation
	TotalAllocatedBytes uint64
	// Peak memory usage in bytes
	PeakMemoryBytes uint64
	// Number of currently active buffers
	ActiveBuffers int64
	// Buffer pool statistics
	PoolAllocated uint64
	PoolReleased  uint64
	PoolHits      uint64
	PoolMisses    uint64
	PooledBuffers int
}

// MemoryStats returns current GPU memory usage statistics.
func (b *Backend) MemoryStats() MemoryStats {
	b.memoryStats.mu.RLock()
	totalAllocated := b.memoryStats.totalAllocatedBytes
	peakMemory := b.memoryStats.peakMemoryBytes
	activeBuffers := b.memoryStats.activeBuffers
	b.memoryStats.mu.RUnlock()

	// Get buffer pool stats
	allocated, released, hits, misses, pooledCount := b.bufferPool.Stats()

	return MemoryStats{
		TotalAllocatedBytes: totalAllocated,
		PeakMemoryBytes:     peakMemory,
		ActiveBuffers:       activeBuffers,
		PoolAllocated:       allocated,
		PoolReleased:        released,
		PoolHits:            hits,
		PoolMisses:          misses,
		PooledBuffers:       pooledCount,
	}
}

// trackBufferAllocation records a buffer allocation in memory statistics.
func (b *Backend) trackBufferAllocation(size uint64) {
	b.memoryStats.mu.Lock()
	defer b.memoryStats.mu.Unlock()

	b.memoryStats.totalAllocatedBytes += size
	b.memoryStats.activeBuffers++

	// Update peak memory if needed
	currentMemory := b.memoryStats.totalAllocatedBytes
	if currentMemory > b.memoryStats.peakMemoryBytes {
		b.memoryStats.peakMemoryBytes = currentMemory
	}
}

// trackBufferRelease records a buffer release in memory statistics.
func (b *Backend) trackBufferRelease(size uint64) {
	b.memoryStats.mu.Lock()
	defer b.memoryStats.mu.Unlock()

	if b.memoryStats.totalAllocatedBytes >= size {
		b.memoryStats.totalAllocatedBytes -= size
	}
	b.memoryStats.activeBuffers--
}

// Gather selects elements along dim using index tensor on GPU.
func (b *Backend) Gather(input *tensor.RawTensor, dim int, indices *tensor.RawTensor) *tensor.RawTensor {
	var result *tensor.RawTensor
	var err error
	if b.LazyMode {
		result, err = b.runGatherLazy(input, dim, indices)
	} else {
		result, err = b.runGather(input, dim, indices)
	}
	if err != nil {
		panic("webgpu: Gather: " + err.Error())
	}
	return result
}

// Where performs conditional element selection on GPU.
// result[i] = condition[i] != 0 ? x[i] : y[i].
func (b *Backend) Where(condition, x, y *tensor.RawTensor) *tensor.RawTensor {
	var result *tensor.RawTensor
	var err error
	if b.LazyMode {
		result, err = b.runWhereLazy(condition, x, y)
	} else {
		result, err = b.runWhere(condition, x, y)
	}
	if err != nil {
		panic("webgpu: Where: " + err.Error())
	}
	return result
}

// Embedding performs embedding lookup on GPU.
// weight: [num_embeddings, embedding_dim], indices: int32 tensor.
// Returns: [...indices_shape, embedding_dim].
func (b *Backend) Embedding(weight, indices *tensor.RawTensor) *tensor.RawTensor {
	result, err := b.runEmbedding(weight, indices)
	if err != nil {
		panic("webgpu: Embedding: " + err.Error())
	}
	return result
}

// ReadGPUBuffer implements tensor.LazyBackend interface.
// Reads data from a GPU staging buffer (MapRead | CopyDst) to CPU memory.
// bufferPtr must point to a *wgpu.Buffer created with BufferUsageMapRead.
//
// The lazy path (runBinaryOpLazy, runUnaryOpLazy, etc.) finishes each encoder
// into a CommandBuffer and queues it in b.pending without submitting. On the
// first Data() call, ReadGPUBuffer flushes the entire pending queue in a single
// queue.Submit, collapsing N individual submits (~500 µs each) into one.
//
// Sequence:
//  1. flushCommands() — submit all pending command buffers in one batch.
//  2. Poll(PollWait) — block until the GPU completes all submitted commands.
//     Required: Map's internal Poll(PollPoll) may return a stale fence when
//     multiple submissions are outstanding (DX12/Vulkan ordering artifact).
//  3. Map the staging buffer (blocks until GPU fence resolves).
//  4. Copy data to CPU slice, Unmap.
func (b *Backend) ReadGPUBuffer(bufferPtr unsafe.Pointer, size uint64) ([]byte, error) {
	b.flushCommands()

	if debugReadGPU {
		fmt.Fprintf(os.Stderr, "[ReadGPUBuffer] Poll(PollWait) start, size=%d\n", size)
	}
	b.device.Poll(wgpu.PollWait)
	if debugReadGPU {
		fmt.Fprintln(os.Stderr, "[ReadGPUBuffer] Poll(PollWait) done")
	}

	buffer := (*wgpu.Buffer)(bufferPtr)

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if debugReadGPU {
		fmt.Fprintln(os.Stderr, "[ReadGPUBuffer] Map start")
	}
	if err := buffer.Map(ctx, wgpu.MapModeRead, 0, size); err != nil {
		return nil, fmt.Errorf("webgpu: ReadGPUBuffer: map failed (size=%d): %w", size, err)
	}
	if debugReadGPU {
		fmt.Fprintln(os.Stderr, "[ReadGPUBuffer] Map done")
	}
	defer func() { _ = buffer.Unmap() }()

	mappedRange, err := buffer.MappedRange(0, size)
	if err != nil {
		return nil, fmt.Errorf("webgpu: ReadGPUBuffer: mapped range (size=%d): %w", size, err)
	}
	defer mappedRange.Release()

	data := mappedRange.Bytes()
	if data == nil {
		return nil, fmt.Errorf("webgpu: ReadGPUBuffer: Bytes() nil (buffer released?)")
	}
	if uint64(len(data)) < size {
		return nil, fmt.Errorf("webgpu: ReadGPUBuffer: got %d bytes, need %d", len(data), size)
	}
	result := make([]byte, size)
	copy(result, data)
	return result, nil
}

// debugReadGPU enables stderr logging for ReadGPUBuffer calls.
// Set via BORN_DEBUG_GPU=1 environment variable.
var debugReadGPU = os.Getenv("BORN_DEBUG_GPU") == "1"

// ReleaseGPUBuffer implements tensor.LazyBackend interface.
// Releases a GPU buffer when no longer needed.
// bufferPtr must be *wgpu.Buffer.
func (b *Backend) ReleaseGPUBuffer(bufferPtr unsafe.Pointer) {
	buffer := (*wgpu.Buffer)(bufferPtr)
	if buffer != nil {
		buffer.Release()
	}
}

// Conv2DInputBackward computes gradient with respect to input for Conv2D.
// Not yet implemented for WebGPU backend.
//
//nolint:revive // Parameters unused in stub implementation.
func (b *Backend) Conv2DInputBackward(input, kernel, grad *tensor.RawTensor, stride, padding int) *tensor.RawTensor {
	panic("webgpu: Conv2DInputBackward not implemented")
}

// Conv2DKernelBackward computes gradient with respect to kernel for Conv2D.
// Not yet implemented for WebGPU backend.
//
//nolint:revive // Parameters unused in stub implementation.
func (b *Backend) Conv2DKernelBackward(input, kernel, grad *tensor.RawTensor, stride, padding int) *tensor.RawTensor {
	panic("webgpu: Conv2DKernelBackward not implemented")
}

// MaxPool2DBackward computes gradient with respect to input for MaxPool2D.
// Not yet implemented for WebGPU backend.
//
//nolint:revive // Parameters unused in stub implementation.
func (b *Backend) MaxPool2DBackward(input, grad *tensor.RawTensor, maxIndices []int, kernelSize, stride int) *tensor.RawTensor {
	panic("webgpu: MaxPool2DBackward not implemented")
}
