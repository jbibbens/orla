// Package telemetry owns the data-plane writers (completion records,
// feedback) and the aggregation queries the mapper reads.
//
// Writes are async-batched via storage.BatchWriter. Reads (Phase 9)
// go through sqlc-generated queries against the same tables. The
// canonical sources are the .sql migrations under internal/storage/
// migrations and the sqlc queries under internal/storage/queries.
package telemetry

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/harvard-cns/orla/internal/storage"
)

// CompletionRecord is one row in the completion_records table.
// Pointers represent NULL-able columns.
//
// LLM dispatches populate PromptTokens / CompletionTokens; tool
// dispatches populate GPUSeconds / ToolKind. Both leave the other
// kind's fields as nil. CostUSD is computed by the proxy at write
// time (token rates × tokens for LLM, gpu_seconds × $/s for tool).
type CompletionRecord struct {
	CompletionID     string            `json:"completion_id"`
	StageID          string            `json:"stage_id"`
	WorkflowRun      string            `json:"workflow_run,omitempty"`
	Backend          string            `json:"backend"`
	Status           string            `json:"status"`
	PromptTokens     *int              `json:"prompt_tokens,omitempty"`
	CompletionTokens *int              `json:"completion_tokens,omitempty"`
	LatencyMs        *int              `json:"latency_ms,omitempty"`
	CostUSD          *float64          `json:"cost_usd,omitempty"`
	GPUSeconds       *float64          `json:"gpu_seconds,omitempty"`
	ToolKind         string            `json:"tool_kind,omitempty"`
	Tags             map[string]string `json:"tags,omitempty"`
	CreatedAt        time.Time         `json:"created_at"`
}

// CompletionWriterConfig is the input to NewCompletionWriter.
type CompletionWriterConfig struct {
	Pool       *pgxpool.Pool
	Logger     *slog.Logger
	BufferSize int           // optional, default 4096
	BatchSize  int           // optional, default 200
	Interval   time.Duration // optional, default 200ms
}

// CompletionWriter is a typed wrapper over storage.BatchWriter for
// completion records. Submit is non-blocking; overflows are dropped
// and counted in Drops.
type CompletionWriter struct {
	bw *storage.BatchWriter[*CompletionRecord]
}

// NewCompletionWriter starts a background flusher that uses pgx.CopyFrom
// to bulk-insert batches into the completion_records table.
func NewCompletionWriter(cfg CompletionWriterConfig) *CompletionWriter {
	if cfg.Logger == nil {
		cfg.Logger = slog.Default()
	}
	bw := storage.NewBatchWriter[*CompletionRecord](storage.BatchWriterConfig[*CompletionRecord]{
		Name:       "completion_records",
		BufferSize: cfg.BufferSize,
		BatchSize:  cfg.BatchSize,
		Interval:   cfg.Interval,
		Flush:      flushCompletions(cfg.Pool),
		Logger:     cfg.Logger,
	})
	return &CompletionWriter{bw: bw}
}

// Submit enqueues a record. Returns false if the writer is closed or
// the buffer is full; either case is counted in Drops.
func (w *CompletionWriter) Submit(rec *CompletionRecord) bool {
	return w.bw.Submit(rec)
}

// Drops returns the cumulative count of dropped records.
func (w *CompletionWriter) Drops() int64 { return w.bw.Drops() }

// Flushes returns the cumulative count of successful flush batches.
func (w *CompletionWriter) Flushes() int64 { return w.bw.Flushes() }

// Failures returns the cumulative count of failed flush attempts.
func (w *CompletionWriter) Failures() int64 { return w.bw.Failures() }

// Close drains the buffer and waits for the final flush, bounded by ctx.
func (w *CompletionWriter) Close(ctx context.Context) error {
	return w.bw.Close(ctx)
}

func flushCompletions(pool *pgxpool.Pool) storage.FlushFunc[*CompletionRecord] {
	columns := []string{
		"completion_id", "stage_id", "workflow_run", "backend", "status",
		"prompt_tokens", "completion_tokens", "latency_ms", "cost_usd",
		"tags", "created_at", "gpu_seconds", "tool_kind",
	}
	return func(ctx context.Context, items []*CompletionRecord) error {
		conn, err := pool.Acquire(ctx)
		if err != nil {
			return fmt.Errorf("acquire conn: %w", err)
		}
		defer conn.Release()

		rows := make([][]any, 0, len(items))
		for _, rec := range items {
			tagsBytes, err := json.Marshal(rec.Tags)
			if err != nil {
				tagsBytes = []byte("{}")
			}
			rows = append(rows, []any{
				rec.CompletionID,
				rec.StageID,
				nullableString(rec.WorkflowRun),
				rec.Backend,
				rec.Status,
				intPtr(rec.PromptTokens),
				intPtr(rec.CompletionTokens),
				intPtr(rec.LatencyMs),
				rec.CostUSD,
				tagsBytes,
				rec.CreatedAt,
				rec.GPUSeconds,
				nullableString(rec.ToolKind),
			})
		}

		_, err = conn.Conn().CopyFrom(ctx,
			pgx.Identifier{"completion_records"},
			columns,
			pgx.CopyFromRows(rows),
		)
		if err != nil {
			return fmt.Errorf("copy completions: %w", err)
		}
		return nil
	}
}

func nullableString(s string) any {
	if s == "" {
		return nil
	}
	return s
}

// intPtr converts *int to int64 pointer for pgx (Postgres INTEGER takes
// int4, but []any with *int requires conversion).
func intPtr(p *int) any {
	if p == nil {
		return nil
	}
	v := int64(*p)
	return v
}
