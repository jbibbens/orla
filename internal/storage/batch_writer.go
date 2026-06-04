package storage

import (
	"context"
	"errors"
	"log/slog"
	"sync"
	"sync/atomic"
	"time"
)

// FlushFunc receives a non-empty batch of items and writes them durably.
// Implementations should return early if ctx is canceled.
type FlushFunc[T any] func(ctx context.Context, items []T) error

// BatchWriterConfig configures a BatchWriter. Zero values for BufferSize,
// BatchSize, and Interval are replaced by defaults.
type BatchWriterConfig[T any] struct {
	// Name identifies the writer in log lines and metrics.
	Name string

	// BufferSize is the channel capacity. Submits past this limit are
	// dropped and counted. Default: 1024.
	BufferSize int

	// BatchSize is the maximum batch size handed to Flush. Default: 100.
	BatchSize int

	// Interval is the maximum delay between flushes. Default: 100ms.
	Interval time.Duration

	// Flush is invoked with a batch of items. Required.
	Flush FlushFunc[T]

	// Logger is used for error reporting. Default: slog.Default().
	Logger *slog.Logger
}

// BatchWriter buffers items in memory and flushes them in batches on
// either a size or time threshold. It is non-blocking: Submit drops the
// item (and increments Drops) when the buffer is full.
//
// Lifecycle: construct with NewBatchWriter, Submit until shutdown, then
// Close. Close stops accepting new items, drains the buffer, and runs
// a final flush.
type BatchWriter[T any] struct {
	name      string
	ch        chan T
	flush     FlushFunc[T]
	batchSize int
	interval  time.Duration
	logger    *slog.Logger

	drops    atomic.Int64
	flushes  atomic.Int64
	failures atomic.Int64

	mu     sync.RWMutex
	closed bool

	cancel context.CancelFunc
	wg     sync.WaitGroup

	closeOnce sync.Once
}

// NewBatchWriter starts the background flusher goroutine. The writer is
// ready to accept Submit calls immediately on return.
//
// Panics if cfg.Flush is nil.
func NewBatchWriter[T any](cfg BatchWriterConfig[T]) *BatchWriter[T] {
	if cfg.Flush == nil {
		panic("storage: BatchWriter requires a non-nil Flush function")
	}
	if cfg.BufferSize <= 0 {
		cfg.BufferSize = 1024
	}
	if cfg.BatchSize <= 0 {
		cfg.BatchSize = 100
	}
	if cfg.Interval <= 0 {
		cfg.Interval = 100 * time.Millisecond
	}
	if cfg.Logger == nil {
		cfg.Logger = slog.Default()
	}

	// cancel is stored on bw and called from Close; lifetime tracked by
	// the struct, not the caller.
	runCtx, cancel := context.WithCancel(context.Background()) //nolint:gosec // see comment above

	bw := &BatchWriter[T]{
		name:      cfg.Name,
		ch:        make(chan T, cfg.BufferSize),
		flush:     cfg.Flush,
		batchSize: cfg.BatchSize,
		interval:  cfg.Interval,
		logger:    cfg.Logger,
		cancel:    cancel,
	}

	bw.wg.Add(1)
	go bw.run(runCtx)
	return bw
}

// Submit attempts to enqueue an item. Returns false (and increments the
// drop counter) if the buffer is full or the writer is closed.
func (b *BatchWriter[T]) Submit(item T) bool {
	b.mu.RLock()
	defer b.mu.RUnlock()
	if b.closed {
		b.drops.Add(1)
		return false
	}
	select {
	case b.ch <- item:
		return true
	default:
		b.drops.Add(1)
		return false
	}
}

// Drops returns the cumulative count of dropped items (buffer full or
// submitted after Close).
func (b *BatchWriter[T]) Drops() int64 { return b.drops.Load() }

// Flushes returns the cumulative count of successful flushes.
func (b *BatchWriter[T]) Flushes() int64 { return b.flushes.Load() }

// Failures returns the cumulative count of failed flush attempts.
func (b *BatchWriter[T]) Failures() int64 { return b.failures.Load() }

// Close stops accepting new Submit calls, drains the buffer, and waits
// for the background flusher to finish. If ctx expires first, the
// in-flight flush is canceled and ctx.Err() is returned. Safe to call
// multiple times; subsequent calls return nil immediately.
func (b *BatchWriter[T]) Close(ctx context.Context) error {
	var err error
	b.closeOnce.Do(func() {
		b.mu.Lock()
		b.closed = true
		close(b.ch)
		b.mu.Unlock()

		done := make(chan struct{})
		go func() {
			b.wg.Wait()
			close(done)
		}()

		select {
		case <-done:
			err = nil
		case <-ctx.Done():
			b.cancel()
			b.wg.Wait()
			err = ctx.Err()
		}
	})
	return err
}

func (b *BatchWriter[T]) run(ctx context.Context) {
	defer b.wg.Done()

	ticker := time.NewTicker(b.interval)
	defer ticker.Stop()

	buf := make([]T, 0, b.batchSize)
	flush := func() {
		if len(buf) == 0 {
			return
		}
		if err := b.flush(ctx, buf); err != nil {
			b.failures.Add(1)
			if !errors.Is(err, context.Canceled) {
				b.logger.Error("batch flush failed",
					"writer", b.name,
					"batch_size", len(buf),
					"error", err,
				)
			}
		} else {
			b.flushes.Add(1)
		}
		buf = buf[:0]
	}

	for {
		select {
		case item, ok := <-b.ch:
			if !ok {
				flush()
				return
			}
			buf = append(buf, item)
			if len(buf) >= b.batchSize {
				flush()
			}
		case <-ticker.C:
			flush()
		case <-ctx.Done():
			// Closer wants to give up. Make one final attempt at items
			// still in the channel without further blocking, then exit.
			for {
				select {
				case item, ok := <-b.ch:
					if ok {
						buf = append(buf, item)
						continue
					}
				default:
				}
				break
			}
			flush()
			return
		}
	}
}
