package nn

import (
	"math/rand"
	"sync"
	"time"
)

// globalRNG is the package-level random number generator for weight initialization.
// When nil, falls back to math/rand global functions.

var (
	// When seeded, we use a single seed to derive worker seeds.
	globalSeed     int64
	globalIsSeeded bool
	globalSeedMu   sync.Mutex

	// A dedicated RNG used solely for deriving seeds for pool workers
	// when a global seed is set.
	seedDeriver *rand.Rand

	// Pool of independent RNGs to avoid lock contention during initialization.
	rngPool = sync.Pool{
		New: func() any {
			globalSeedMu.Lock()
			seeded := globalIsSeeded
			var derivedSeed int64
			if seeded && seedDeriver != nil {
				derivedSeed = seedDeriver.Int63()
			}
			globalSeedMu.Unlock()

			if seeded {
				// If global seed is set, derive a seed for this new worker RNG
				return rand.New(rand.NewSource(derivedSeed)) //nolint:gosec // ML uses math/rand intentionally
			}
			return rand.New(rand.NewSource(time.Now().UnixNano())) //nolint:gosec // ML uses math/rand intentionally
		},
	}
)

// SetSeed sets the global random seed for ALL random operations:
// weight initialization (Xavier, Embedding) AND tensor creation (Randn, Rand).
//
// Call this before creating models to ensure reproducible initialization.
// Without calling SetSeed, initialization uses Go's auto-seeded global rand
// (non-deterministic across runs).
//
// Thread-safe: can be called concurrently with model creation.
//
// Example:
//
//	nn.SetSeed(42)
//	model1 := NewLinear(784, 128, backend)
//
//	nn.SetSeed(42)
//	model2 := NewLinear(784, 128, backend) // identical weights to model1

// SetSeed sets the global random seed for tensor random operations (Randn, Rand).
//
// Call this before creating random tensors to ensure reproducible results.
// Thread-safe.
func SetSeed(seed int64) {
	globalSeedMu.Lock()
	defer globalSeedMu.Unlock()
	globalSeed = seed
	globalIsSeeded = true
	// Use a private, lock-protected RNG to derive deterministic seeds for pool workers.
	seedDeriver = rand.New(rand.NewSource(seed))
}

// ResetSeed clears the seeded RNG, reverting to Go's auto-seeded global rand.
func ResetSeed() {
	globalSeedMu.Lock()
	defer globalSeedMu.Unlock()
	globalIsSeeded = false
	seedDeriver = nil
}

// RandFloat64 returns a random float64 from an independent RNG pool.
// Avoids global lock contention during concurrent model initialization.
func RandFloat64() float64 {
	r := rngPool.Get().(*rand.Rand)
	val := r.Float64()
	rngPool.Put(r)
	return val
}

// randNormFloat64 returns a random float64 from standard normal distribution.
// Avoids global lock contention during concurrent model initialization.
func randNormFloat64() float64 {
	r := rngPool.Get().(*rand.Rand)
	val := r.NormFloat64()
	rngPool.Put(r)
	return val
}
