package storage

import (
	"context"
	"errors"
	"io"
	"log/slog"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// testLogger returns a logger that discards output, useful when we
// expect intentional flush errors.
func testLogger() *slog.Logger {
	return slog.New(slog.NewTextHandler(io.Discard, nil))
}

// waitFor polls fn every 5ms up to timeout. Returns true if fn ever
// returns true. Lets tests wait on async background work without
// hard-coding sleeps.
func waitFor(t *testing.T, timeout time.Duration, fn func() bool) bool {
	t.Helper()
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if fn() {
			return true
		}
		time.Sleep(5 * time.Millisecond)
	}
	return fn()
}

func TestBatchWriter_FlushesOnBatchSize(t *testing.T) {
	var captured [][]int
	var mu sync.Mutex
	bw := NewBatchWriter[int](BatchWriterConfig[int]{
		Name:       "size",
		BufferSize: 100,
		BatchSize:  3,
		Interval:   10 * time.Second, // effectively disabled
		Flush: func(_ context.Context, items []int) error {
			mu.Lock()
			defer mu.Unlock()
			out := make([]int, len(items))
			copy(out, items)
			captured = append(captured, out)
			return nil
		},
		Logger: testLogger(),
	})
	t.Cleanup(func() { _ = bw.Close(context.Background()) })

	bw.Submit(1)
	bw.Submit(2)
	bw.Submit(3) // triggers flush

	require.True(t, waitFor(t, time.Second, func() bool {
		return bw.Flushes() == 1
	}), "expected 1 flush, got %d", bw.Flushes())

	mu.Lock()
	defer mu.Unlock()
	require.Len(t, captured, 1)
	assert.Equal(t, []int{1, 2, 3}, captured[0])
}

func TestBatchWriter_FlushesOnInterval(t *testing.T) {
	var captured [][]int
	var mu sync.Mutex
	bw := NewBatchWriter[int](BatchWriterConfig[int]{
		Name:       "interval",
		BufferSize: 100,
		BatchSize:  100, // intentionally larger than what we submit
		Interval:   20 * time.Millisecond,
		Flush: func(_ context.Context, items []int) error {
			mu.Lock()
			defer mu.Unlock()
			out := make([]int, len(items))
			copy(out, items)
			captured = append(captured, out)
			return nil
		},
		Logger: testLogger(),
	})
	t.Cleanup(func() { _ = bw.Close(context.Background()) })

	bw.Submit(7)

	require.True(t, waitFor(t, time.Second, func() bool {
		return bw.Flushes() >= 1
	}), "expected at least 1 flush")

	mu.Lock()
	defer mu.Unlock()
	require.NotEmpty(t, captured)
	assert.Equal(t, []int{7}, captured[0])
}

func TestBatchWriter_CloseDrainsBuffer(t *testing.T) {
	var total atomic.Int32
	bw := NewBatchWriter[int](BatchWriterConfig[int]{
		Name:       "drain",
		BufferSize: 100,
		BatchSize:  10,
		Interval:   10 * time.Second,
		Flush: func(_ context.Context, items []int) error {
			total.Add(int32(len(items)))
			return nil
		},
		Logger: testLogger(),
	})

	for i := range 5 {
		require.True(t, bw.Submit(i))
	}

	require.NoError(t, bw.Close(context.Background()))
	assert.Equal(t, int32(5), total.Load(), "Close must flush remaining items")
}

func TestBatchWriter_DropsWhenBufferFull(t *testing.T) {
	gate := make(chan struct{})
	bw := NewBatchWriter[int](BatchWriterConfig[int]{
		Name:       "drops",
		BufferSize: 2,
		BatchSize:  1,
		Interval:   10 * time.Second,
		Flush: func(_ context.Context, _ []int) error {
			<-gate // hold the flusher so the buffer can fill up
			return nil
		},
		Logger: testLogger(),
	})
	t.Cleanup(func() {
		close(gate)
		_ = bw.Close(context.Background())
	})

	// First Submit kicks the flusher (which blocks on gate); subsequent
	// submits fill the BufferSize=2 channel and then overflow.
	require.True(t, bw.Submit(1))

	// Wait until the run goroutine has taken the first item off the
	// channel, otherwise we may not actually fill the buffer.
	time.Sleep(20 * time.Millisecond)

	require.True(t, bw.Submit(2))
	require.True(t, bw.Submit(3))
	assert.False(t, bw.Submit(4), "expected drop")
	assert.False(t, bw.Submit(5), "expected drop")

	assert.GreaterOrEqual(t, bw.Drops(), int64(2))
}

func TestBatchWriter_SubmitAfterCloseDrops(t *testing.T) {
	bw := NewBatchWriter[int](BatchWriterConfig[int]{
		Name:       "after-close",
		BufferSize: 4,
		BatchSize:  4,
		Interval:   10 * time.Second,
		Flush:      func(_ context.Context, _ []int) error { return nil },
		Logger:     testLogger(),
	})
	require.NoError(t, bw.Close(context.Background()))

	assert.False(t, bw.Submit(1))
	assert.Equal(t, int64(1), bw.Drops())
}

func TestBatchWriter_CloseIdempotent(t *testing.T) {
	bw := NewBatchWriter[int](BatchWriterConfig[int]{
		Name:   "idempotent",
		Flush:  func(_ context.Context, _ []int) error { return nil },
		Logger: testLogger(),
	})
	require.NoError(t, bw.Close(context.Background()))
	assert.NoError(t, bw.Close(context.Background()))
}

func TestBatchWriter_FlushFailureCounted(t *testing.T) {
	bw := NewBatchWriter[int](BatchWriterConfig[int]{
		Name:       "fail",
		BufferSize: 4,
		BatchSize:  1,
		Interval:   10 * time.Second,
		Flush: func(_ context.Context, _ []int) error {
			return errors.New("simulated failure")
		},
		Logger: testLogger(),
	})
	t.Cleanup(func() { _ = bw.Close(context.Background()) })

	bw.Submit(1)
	require.True(t, waitFor(t, time.Second, func() bool {
		return bw.Failures() == 1
	}))
	assert.Equal(t, int64(0), bw.Flushes())
}

func TestBatchWriter_NilFlushPanics(t *testing.T) {
	assert.Panics(t, func() {
		NewBatchWriter[int](BatchWriterConfig[int]{Name: "panic"})
	})
}

func TestBatchWriter_CloseTimeoutReturnsCtxErr(t *testing.T) {
	stuck := make(chan struct{})
	bw := NewBatchWriter[int](BatchWriterConfig[int]{
		Name:       "stuck",
		BufferSize: 4,
		BatchSize:  1,
		Interval:   10 * time.Second,
		Flush: func(ctx context.Context, _ []int) error {
			select {
			case <-stuck:
			case <-ctx.Done():
			}
			return ctx.Err()
		},
		Logger: testLogger(),
	})
	t.Cleanup(func() { close(stuck) })

	bw.Submit(1)
	time.Sleep(20 * time.Millisecond) // ensure flush starts

	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()
	err := bw.Close(ctx)
	require.Error(t, err)
	assert.True(t, errors.Is(err, context.DeadlineExceeded))
}
