// Package tolerance provides helpers for approximate floating-point equality
// using absolute, relative, or combined tolerance checks.
package tolerance

import (
	"fmt"
	"math"
)

// TolType specifies the kind of tolerance check to apply.
type TolType int

const (
	// RelAbs uses both absolute and relative tolerance, passing if either check passes (default).
	RelAbs TolType = iota
	// Rel uses only relative tolerance.
	Rel
	// Abs uses only absolute tolerance.
	Abs
)

// Tolerance holds parameters for approximate floating-point equality checks.
//
// The zero value is not usable; construct one with NewDefaultTolerance or
// a struct literal with exported fields.
type Tolerance[T float32 | float64] struct {
	tolType TolType
	abs     T // absolute tolerance
	rel     T // relative tolerance factor
}

// NewDefaultTolerance returns a Tolerance with sensible defaults.
//
// The tolerance type is set to RelAbs with the following values:
//   - float32: rel = 1e-2, abs = 1e-5
//   - float64: rel = 1e-4, abs = 1e-5
func NewDefaultTolerance[T float32 | float64]() *Tolerance[T] {
	var rel T
	switch any(rel).(type) {
	case float32:
		rel = 1e-2
	case float64:
		rel = 1e-4
	}
	return &Tolerance[T]{
		tolType: RelAbs,
		abs:     1e-5,
		rel:     rel,
	}
}

// AssertApproxEqual checks whether a and b are approximately equal according
// to the given tolerance.
//
// Absolute tolerance: passes when |a-b| < tol.abs.
// Relative tolerance: passes when |a-b| < tol.rel * |a+b|.
// Combined (RelAbs): passes when either condition is met.
//
// Returns nil if the values are approximately equal, or an error describing
// which tolerance check failed.
func AssertApproxEqual[T float32 | float64](a, b T, tol *Tolerance[T]) error {
	switch tol.tolType {
	case Abs:
		return checkAbsolute(a, b, tol.abs)
	case Rel:
		return checkRelative(a, b, tol.rel)
	case RelAbs:
		absErr := checkAbsolute(a, b, tol.abs)
		relErr := checkRelative(a, b, tol.rel)
		switch {
		case absErr == nil || relErr == nil:
			return nil
		case absErr != nil:
			return absErr
		case relErr != nil:
			return relErr
		}
	}
	return nil
}

// AssertAllApproxEqual checks a and b element-wise for approximate equality.
//
// Returns an error describing the first element that fails, or a length
// mismatch error if the slices differ in length.
func AssertAllApproxEqual[T float32 | float64](a, b []T, tol *Tolerance[T]) error {
	if len(a) != len(b) {
		return fmt.Errorf("slices differ in length: len(a)=%d; len(b)=%d", len(a), len(b))
	}
	for i := range a {
		if err := AssertApproxEqual(a[i], b[i], tol); err != nil {
			return fmt.Errorf("element at index %d: %w", i, err)
		}
	}
	return nil
}

// checkRelative is a helper to compare two values using absolute tolerance.
//
// Returns nil if |a-b| < tol.rel * |a+b|, an error otherwise.
func checkRelative[T float32 | float64](a, b, rel T) error {
	absDiff := math.Abs(float64(a - b))
	relTol := float64(rel) * math.Abs(float64(a+b))
	// handles NaN case, if a or b is NaN then absDiff compared to anything will be false
	if absDiff < relTol {
		return nil
	}
	return fmt.Errorf("relative tolerance failure: %f >= %f", absDiff, relTol)
}

// checkAbsolute is a helper to compare two values using absolute tolerance.
//
// Returns nil if |a-b| < tol.abs, an error otherwise.
func checkAbsolute[T float32 | float64](a, b, abs T) error {
	absDiff := math.Abs(float64(a - b))
	// handles NaN case, if a or b is NaN then absDiff compared to anything will be false
	if absDiff < float64(abs) {
		return nil
	}
	return fmt.Errorf("absolute tolerance failure: %f >= %f", absDiff, abs)
}
