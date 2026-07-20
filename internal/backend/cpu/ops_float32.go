package cpu

import (
	"runtime"

	"github.com/xucanxx/born/internal/parallel"
	"github.com/xucanxx/born/internal/tensor"
)

// Float32 inplace operations

func addInplaceFloat32(a, b []float32) {
	if simdAddInplaceFloat32 != nil {
		simdAddInplaceFloat32(a, b)
		return
	}
	for i := range a {
		a[i] += b[i]
	}
}

func subInplaceFloat32(a, b []float32) {
	if simdSubInplaceFloat32 != nil {
		simdSubInplaceFloat32(a, b)
		return
	}
	for i := range a {
		a[i] -= b[i]
	}
}

func mulInplaceFloat32(a, b []float32) {
	if simdMulInplaceFloat32 != nil {
		simdMulInplaceFloat32(a, b)
		return
	}
	for i := range a {
		a[i] *= b[i]
	}
}

func divInplaceFloat32(a, b []float32) {
	if simdDivInplaceFloat32 != nil {
		simdDivInplaceFloat32(a, b)
		return
	}
	for i := range a {
		a[i] /= b[i]
	}
}

// Float32 vectorized operations

func addVectorizedFloat32(dst, a, b []float32) {
	if simdAddVectorizedFloat32 != nil {
		simdAddVectorizedFloat32(dst, a, b)
		return
	}
	for i := range a {
		dst[i] = a[i] + b[i]
	}
}

func subVectorizedFloat32(dst, a, b []float32) {
	if simdSubVectorizedFloat32 != nil {
		simdSubVectorizedFloat32(dst, a, b)
		return
	}
	for i := range a {
		dst[i] = a[i] - b[i]
	}
}

func mulVectorizedFloat32(dst, a, b []float32) {
	if simdMulVectorizedFloat32 != nil {
		simdMulVectorizedFloat32(dst, a, b)
		return
	}
	for i := range a {
		dst[i] = a[i] * b[i]
	}
}

func divVectorizedFloat32(dst, a, b []float32) {
	if simdDivVectorizedFloat32 != nil {
		simdDivVectorizedFloat32(dst, a, b)
		return
	}
	for i := range a {
		dst[i] = a[i] / b[i]
	}
}

// Float32 broadcasting operations

// The broadcasting ops below write a contiguous, row-major dst, so instead of
// recomputing each source index with computeFlatIndex (an integer division and
// modulo per output dimension, for every element) they advance the source flat
// indices incrementally: a mixed-radix odometer over the output coordinates
// updates aIdx and bIdx with a couple of adds per step and a carry only at
// dimension boundaries. The result is bit-identical to the division form (the
// same operands combined in the same order).

func addBroadcastFloat32(dst, a, b []float32, aShape, bShape, outShape tensor.Shape) {
	aStrides := computeBroadcastStridesForShape(aShape, outShape)
	bStrides := computeBroadcastStridesForShape(bShape, outShape)

	ndim := len(outShape)

	switch ndim {
	case 2:
		batch := outShape[0]
		channels := outShape[1]
		as0, as1 := aStrides[0], aStrides[1]
		bs0, bs1 := bStrides[0], bStrides[1]

		parallel.ForBatch(batch, channels, func(c0, c1 int) {
			aIdx := c0*as0 + c1*as1
			bIdx := c0*bs0 + c1*bs1
			dst[c0*channels+c1] = a[aIdx] + b[bIdx]
		}, parallel.Config{Enabled: true, NumWorkers: runtime.NumCPU(), MinChunkSize: 2048})

	case 3:
		batch := outShape[0]
		channels := outShape[1] * outShape[2]
		as0, as1, as2 := aStrides[0], aStrides[1], aStrides[2]
		bs0, bs1, bs2 := bStrides[0], bStrides[1], bStrides[2]
		s2 := outShape[2]

		parallel.ForBatch(batch, channels, func(c0, rem int) {
			c1 := rem / s2
			c2 := rem % s2

			aIdx := c0*as0 + c1*as1 + c2*as2
			bIdx := c0*bs0 + c1*bs1 + c2*bs2
			dst[c0*channels+rem] = a[aIdx] + b[bIdx]
		}, parallel.Config{Enabled: true, NumWorkers: runtime.NumCPU(), MinChunkSize: 2048})

	case 4:
		batch := outShape[0] * outShape[1]
		channels := outShape[2] * outShape[3]
		as0, as1, as2, as3 := aStrides[0], aStrides[1], aStrides[2], aStrides[3]
		bs0, bs1, bs2, bs3 := bStrides[0], bStrides[1], bStrides[2], bStrides[3]
		s1 := outShape[1]
		s3 := outShape[3]

		parallel.ForBatch(batch, channels, func(b_, c_ int) {
			c0 := b_ / s1
			c1 := b_ % s1
			c2 := c_ / s3
			c3 := c_ % s3

			aIdx := c0*as0 + c1*as1 + c2*as2 + c3*as3
			bIdx := c0*bs0 + c1*bs1 + c2*bs2 + c3*bs3
			dst[b_*channels+c_] = a[aIdx] + b[bIdx]
		}, parallel.Config{Enabled: true, NumWorkers: runtime.NumCPU(), MinChunkSize: 2048})

	default:
		n := outShape.NumElements()
		outStrides := outShape.ComputeStrides()
		parallel.For(n, func(i int) {
			var coords [8]int
			idx := i
			for dim := 0; dim < ndim; dim++ {
				coords[dim] = idx / outStrides[dim]
				idx %= outStrides[dim]
			}

			aIdx := 0
			bIdx := 0
			for dim := 0; dim < ndim; dim++ {
				aIdx += coords[dim] * aStrides[dim]
				bIdx += coords[dim] * bStrides[dim]
			}
			dst[i] = a[aIdx] + b[bIdx]
		}, parallel.Config{Enabled: true, NumWorkers: runtime.NumCPU(), MinChunkSize: 2048})
	}
}

func subBroadcastFloat32(dst, a, b []float32, aShape, bShape, outShape tensor.Shape) {
	aStrides := computeBroadcastStridesForShape(aShape, outShape)
	bStrides := computeBroadcastStridesForShape(bShape, outShape)

	ndim := len(outShape)

	switch ndim {
	case 2:
		batch := outShape[0]
		channels := outShape[1]
		as0, as1 := aStrides[0], aStrides[1]
		bs0, bs1 := bStrides[0], bStrides[1]

		parallel.ForBatch(batch, channels, func(c0, c1 int) {
			aIdx := c0*as0 + c1*as1
			bIdx := c0*bs0 + c1*bs1
			dst[c0*channels+c1] = a[aIdx] - b[bIdx]
		}, parallel.Config{Enabled: true, NumWorkers: runtime.NumCPU(), MinChunkSize: 2048})

	case 3:
		batch := outShape[0]
		channels := outShape[1] * outShape[2]
		as0, as1, as2 := aStrides[0], aStrides[1], aStrides[2]
		bs0, bs1, bs2 := bStrides[0], bStrides[1], bStrides[2]
		s2 := outShape[2]

		parallel.ForBatch(batch, channels, func(c0, rem int) {
			c1 := rem / s2
			c2 := rem % s2

			aIdx := c0*as0 + c1*as1 + c2*as2
			bIdx := c0*bs0 + c1*bs1 + c2*bs2
			dst[c0*channels+rem] = a[aIdx] - b[bIdx]
		}, parallel.Config{Enabled: true, NumWorkers: runtime.NumCPU(), MinChunkSize: 2048})

	case 4:
		batch := outShape[0] * outShape[1]
		channels := outShape[2] * outShape[3]
		as0, as1, as2, as3 := aStrides[0], aStrides[1], aStrides[2], aStrides[3]
		bs0, bs1, bs2, bs3 := bStrides[0], bStrides[1], bStrides[2], bStrides[3]
		s1 := outShape[1]
		s3 := outShape[3]

		parallel.ForBatch(batch, channels, func(b_, c_ int) {
			c0 := b_ / s1
			c1 := b_ % s1
			c2 := c_ / s3
			c3 := c_ % s3

			aIdx := c0*as0 + c1*as1 + c2*as2 + c3*as3
			bIdx := c0*bs0 + c1*bs1 + c2*bs2 + c3*bs3
			dst[b_*channels+c_] = a[aIdx] - b[bIdx]
		}, parallel.Config{Enabled: true, NumWorkers: runtime.NumCPU(), MinChunkSize: 2048})

	default:
		n := outShape.NumElements()
		outStrides := outShape.ComputeStrides()
		parallel.For(n, func(i int) {
			var coords [8]int
			idx := i
			for dim := 0; dim < ndim; dim++ {
				coords[dim] = idx / outStrides[dim]
				idx %= outStrides[dim]
			}

			aIdx := 0
			bIdx := 0
			for dim := 0; dim < ndim; dim++ {
				aIdx += coords[dim] * aStrides[dim]
				bIdx += coords[dim] * bStrides[dim]
			}
			dst[i] = a[aIdx] - b[bIdx]
		}, parallel.Config{Enabled: true, NumWorkers: runtime.NumCPU(), MinChunkSize: 2048})
	}
}

func mulBroadcastFloat32(dst, a, b []float32, aShape, bShape, outShape tensor.Shape) {
	// Fast paths for the structured broadcasts the model uses: one operand spans the
	// full output and the other either is constant over a trailing run (per-channel
	// scale, e.g. [N,C,H,W]*[N,C,1,1]) or is a dense vector tiled over the leading
	// dims (per-feature scale, e.g. [M,L]*[L], the STFT front end). Both avoid the
	// per-element index odometer. Multiply commutes, so try either operand as full.
	if mulBroadcastFullFloat32(dst, a, b, aShape, bShape, outShape) ||
		mulBroadcastFullFloat32(dst, b, a, bShape, aShape, outShape) {
		return
	}

	aStrides := computeBroadcastStridesForShape(aShape, outShape)
	bStrides := computeBroadcastStridesForShape(bShape, outShape)

	ndim := len(outShape)

	switch ndim {
	case 2:
		batch := outShape[0]
		channels := outShape[1]
		as0, as1 := aStrides[0], aStrides[1]
		bs0, bs1 := bStrides[0], bStrides[1]

		parallel.ForBatch(batch, channels, func(c0, c1 int) {
			aIdx := c0*as0 + c1*as1
			bIdx := c0*bs0 + c1*bs1
			dst[c0*channels+c1] = a[aIdx] * b[bIdx]
		}, parallel.Config{Enabled: true, NumWorkers: runtime.NumCPU(), MinChunkSize: 2048})

	case 3:
		batch := outShape[0]
		channels := outShape[1] * outShape[2]
		as0, as1, as2 := aStrides[0], aStrides[1], aStrides[2]
		bs0, bs1, bs2 := bStrides[0], bStrides[1], bStrides[2]
		s2 := outShape[2]

		parallel.ForBatch(batch, channels, func(c0, rem int) {
			c1 := rem / s2
			c2 := rem % s2

			aIdx := c0*as0 + c1*as1 + c2*as2
			bIdx := c0*bs0 + c1*bs1 + c2*bs2
			dst[c0*channels+rem] = a[aIdx] * b[bIdx]
		}, parallel.Config{Enabled: true, NumWorkers: runtime.NumCPU(), MinChunkSize: 2048})

	case 4:
		batch := outShape[0] * outShape[1]
		channels := outShape[2] * outShape[3]
		as0, as1, as2, as3 := aStrides[0], aStrides[1], aStrides[2], aStrides[3]
		bs0, bs1, bs2, bs3 := bStrides[0], bStrides[1], bStrides[2], bStrides[3]
		s1 := outShape[1]
		s3 := outShape[3]

		parallel.ForBatch(batch, channels, func(b_, c_ int) {
			c0 := b_ / s1
			c1 := b_ % s1
			c2 := c_ / s3
			c3 := c_ % s3

			aIdx := c0*as0 + c1*as1 + c2*as2 + c3*as3
			bIdx := c0*bs0 + c1*bs1 + c2*bs2 + c3*bs3
			dst[b_*channels+c_] = a[aIdx] * b[bIdx]
		}, parallel.Config{Enabled: true, NumWorkers: runtime.NumCPU(), MinChunkSize: 2048})

	default:
		n := outShape.NumElements()
		outStrides := outShape.ComputeStrides()
		parallel.For(n, func(i int) {
			var coords [8]int
			idx := i
			for dim := 0; dim < ndim; dim++ {
				coords[dim] = idx / outStrides[dim]
				idx %= outStrides[dim]
			}

			aIdx := 0
			bIdx := 0
			for dim := 0; dim < ndim; dim++ {
				aIdx += coords[dim] * aStrides[dim]
				bIdx += coords[dim] * bStrides[dim]
			}
			dst[i] = a[aIdx] * b[bIdx]
		}, parallel.Config{Enabled: true, NumWorkers: runtime.NumCPU(), MinChunkSize: 2048})
	}
}

// mulBroadcastFullFloat32 computes dst = full * bc when `full` spans the whole
// output and bc broadcasts in one of two structured ways (trailing-run scale or
// dense leading-tiled vector). It returns false without writing otherwise, so the
// caller can try the commuted order or fall back to the general odometer.
func mulBroadcastFullFloat32(dst, full, bc []float32, fullShape, bcShape, outShape tensor.Shape) bool {
	if !fullShape.Equal(outShape) {
		return false
	}
	bcStrides := computeBroadcastStridesForShape(bcShape, outShape)
	// Iterate over the logical element count, not len(dst): the backing slice may
	// carry slack beyond the tensor's shape, and walking it would read/write past
	// the valid region.
	n := outShape.NumElements()

	// Case A: bc is constant over a trailing run (zeros at the tail of bcStrides),
	// so each run is a scalar multiply.
	if run := trailingBroadcastRun(bcStrides, outShape); run > 1 {
		outStrides := outShape.ComputeStrides()
		parallel.For((n+run-1)/run, func(i int) {
			baseIndex := i * run
			s := bc[computeFlatIndex(baseIndex, outStrides, bcStrides)]

			endIndex := baseIndex + run
			if endIndex > n {
				endIndex = n
			}

			d := dst[baseIndex:endIndex]
			f := full[baseIndex:endIndex]
			for j := range d {
				d[j] = f[j] * s
			}
		}, parallel.Config{Enabled: true, NumWorkers: runtime.NumCPU(), MinChunkSize: 1})
		return true
	}

	// Case B: bc is a dense trailing tile, constant over the leading dims, so
	// dst[i] = full[i] * bc[i % tile] (each full row times the bc vector).
	if tile := leadingBroadcastTile(bcStrides, bcShape, outShape); tile > 0 {
		bct := bc[:tile]
		for base := 0; base < n; base += tile {
			d := dst[base : base+tile]
			f := full[base : base+tile]
			for j, bv := range bct {
				d[j] = f[j] * bv
			}
		}
		return true
	}
	return false
}

// trailingBroadcastRun returns the product of outShape's trailing dimensions over
// which bStrides is 0 (the operand is constant there), i.e. the largest contiguous
// run of the output that maps to a single source element. Returns 1 when the last
// dimension is not broadcast.
func trailingBroadcastRun(bStrides []int, outShape tensor.Shape) int {
	run := 1
	for d := len(outShape) - 1; d >= 0; d-- {
		if bStrides[d] != 0 {
			break
		}
		run *= outShape[d]
	}
	return run
}

// leadingBroadcastTile returns the dense trailing-tile length when bc is constant
// over the leading output dims and dense (its natural strides) over a contiguous
// trailing tile, so dst[i] = full[i] * bc[i % tile]. Returns 0 if the pattern does
// not hold.
func leadingBroadcastTile(bcStrides []int, bcShape, outShape tensor.Shape) int {
	ndim := len(outShape)
	split := 0
	for split < ndim && bcStrides[split] == 0 {
		split++
	}
	if split == 0 || split == ndim {
		return 0 // no leading broadcast, or fully broadcast (the trailing-run case)
	}
	outStrides := outShape.ComputeStrides()
	tile := 1
	for d := split; d < ndim; d++ {
		if bcStrides[d] != outStrides[d] {
			return 0
		}
		tile *= outShape[d]
	}
	if tile != bcShape.NumElements() {
		return 0
	}
	return tile
}

func divBroadcastFloat32(dst, a, b []float32, aShape, bShape, outShape tensor.Shape) {
	aStrides := computeBroadcastStridesForShape(aShape, outShape)
	bStrides := computeBroadcastStridesForShape(bShape, outShape)

	ndim := len(outShape)

	switch ndim {
	case 2:
		batch := outShape[0]
		channels := outShape[1]
		as0, as1 := aStrides[0], aStrides[1]
		bs0, bs1 := bStrides[0], bStrides[1]

		parallel.ForBatch(batch, channels, func(c0, c1 int) {
			aIdx := c0*as0 + c1*as1
			bIdx := c0*bs0 + c1*bs1
			dst[c0*channels+c1] = a[aIdx] / b[bIdx]
		}, parallel.Config{Enabled: true, NumWorkers: runtime.NumCPU(), MinChunkSize: 2048})

	case 3:
		batch := outShape[0]
		channels := outShape[1] * outShape[2]
		as0, as1, as2 := aStrides[0], aStrides[1], aStrides[2]
		bs0, bs1, bs2 := bStrides[0], bStrides[1], bStrides[2]
		s2 := outShape[2]

		parallel.ForBatch(batch, channels, func(c0, rem int) {
			c1 := rem / s2
			c2 := rem % s2

			aIdx := c0*as0 + c1*as1 + c2*as2
			bIdx := c0*bs0 + c1*bs1 + c2*bs2
			dst[c0*channels+rem] = a[aIdx] / b[bIdx]
		}, parallel.Config{Enabled: true, NumWorkers: runtime.NumCPU(), MinChunkSize: 2048})

	case 4:
		batch := outShape[0] * outShape[1]
		channels := outShape[2] * outShape[3]
		as0, as1, as2, as3 := aStrides[0], aStrides[1], aStrides[2], aStrides[3]
		bs0, bs1, bs2, bs3 := bStrides[0], bStrides[1], bStrides[2], bStrides[3]
		s1 := outShape[1]
		s3 := outShape[3]

		parallel.ForBatch(batch, channels, func(b_, c_ int) {
			c0 := b_ / s1
			c1 := b_ % s1
			c2 := c_ / s3
			c3 := c_ % s3

			aIdx := c0*as0 + c1*as1 + c2*as2 + c3*as3
			bIdx := c0*bs0 + c1*bs1 + c2*bs2 + c3*bs3
			dst[b_*channels+c_] = a[aIdx] / b[bIdx]
		}, parallel.Config{Enabled: true, NumWorkers: runtime.NumCPU(), MinChunkSize: 2048})

	default:
		n := outShape.NumElements()
		outStrides := outShape.ComputeStrides()
		parallel.For(n, func(i int) {
			var coords [8]int
			idx := i
			for dim := 0; dim < ndim; dim++ {
				coords[dim] = idx / outStrides[dim]
				idx %= outStrides[dim]
			}

			aIdx := 0
			bIdx := 0
			for dim := 0; dim < ndim; dim++ {
				aIdx += coords[dim] * aStrides[dim]
				bIdx += coords[dim] * bStrides[dim]
			}
			dst[i] = a[aIdx] / b[bIdx]
		}, parallel.Config{Enabled: true, NumWorkers: runtime.NumCPU(), MinChunkSize: 2048})
	}
}

// Transpose float32.
func transposeFloat32(dst, src []float32, shape tensor.Shape, axes []int) {
	ndim := len(shape)
	srcStrides := shape.ComputeStrides()

	// Compute destination shape and strides
	dstShape := make(tensor.Shape, ndim)
	for i, ax := range axes {
		dstShape[i] = shape[ax]
	}
	dstStrides := dstShape.ComputeStrides()

	n := shape.NumElements()

	switch ndim {
	case 2:
		ds0, ds1 := dstStrides[0], dstStrides[1]
		ax0, ax1 := axes[0], axes[1]
		channels := shape[1]

		// 2D 使用 ForBatch: c0 是行, c1 是列
		parallel.ForBatch(shape[0], channels, func(c0, c1 int) {
			var p0, p1 int
			if ax0 == 0 {
				p0 = c0
			} else {
				p0 = c1
			}
			if ax1 == 0 {
				p1 = c0
			} else {
				p1 = c1
			}
			// c0*channels+c1 等价于一维平坦索引 i
			dst[p0*ds0+p1*ds1] = src[c0*channels+c1]
		}, parallel.Config{Enabled: true, NumWorkers: runtime.NumCPU(), MinChunkSize: 2048})

	case 3:
		ds0, ds1, ds2 := dstStrides[0], dstStrides[1], dstStrides[2]
		ax0, ax1, ax2 := axes[0], axes[1], axes[2]
		channels := shape[1] * shape[2]

		// 3D 把后两维压扁成 channels，ForBatch 返回 c0 和 rem
		parallel.ForBatch(shape[0], channels, func(c0, rem int) {
			c1 := rem / shape[2]
			c2 := rem % shape[2]

			var coords [3]int
			coords[0], coords[1], coords[2] = c0, c1, c2

			dstIdx := coords[ax0]*ds0 + coords[ax1]*ds1 + coords[ax2]*ds2
			dst[dstIdx] = src[c0*channels+rem]
		}, parallel.Config{Enabled: true, NumWorkers: runtime.NumCPU(), MinChunkSize: 2048})

	case 4:
		ds0, ds1, ds2, ds3 := dstStrides[0], dstStrides[1], dstStrides[2], dstStrides[3]
		ax0, ax1, ax2, ax3 := axes[0], axes[1], axes[2], axes[3]
		batch := shape[0] * shape[1]
		channels := shape[2] * shape[3]

		// 4D 把前两维和后两维分别压扁
		parallel.ForBatch(batch, channels, func(b, c int) {
			c0 := b / shape[1]
			c1 := b % shape[1]
			c2 := c / shape[3]
			c3 := c % shape[3]

			var coords [4]int
			coords[0], coords[1], coords[2], coords[3] = c0, c1, c2, c3

			dstIdx := coords[ax0]*ds0 + coords[ax1]*ds1 + coords[ax2]*ds2 + coords[ax3]*ds3
			dst[dstIdx] = src[b*channels+c]
		}, parallel.Config{Enabled: true, NumWorkers: runtime.NumCPU(), MinChunkSize: 2048})

	default:
		// 通用降级路径 (ndim > 4)
		parallel.For(n, func(i int) {
			var coords [8]int
			idx := i
			for dim := 0; dim < ndim; dim++ {
				coords[dim] = idx / srcStrides[dim]
				idx %= srcStrides[dim]
			}

			dstIdx := 0
			for dstDim, srcDim := range axes {
				dstIdx += coords[srcDim] * dstStrides[dstDim]
			}
			dst[dstIdx] = src[i]
		}, parallel.Config{Enabled: true, NumWorkers: runtime.NumCPU(), MinChunkSize: 2048})
	}
}
