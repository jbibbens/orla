// Package storage opens the Postgres connection pool, runs goose
// migrations, and exposes both a pgx-native pool (for sqlc-generated
// queries) and a *sql.DB adapter (for migrations and any future code
// that prefers database/sql).
package storage

import (
	"context"
	"database/sql"
	"embed"
	"errors"
	"fmt"
	"io/fs"
	"log/slog"
	"math"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/jackc/pgx/v5/stdlib"
	"github.com/pressly/goose/v3"
)

//go:embed migrations/*.sql
var migrationsFS embed.FS

// OpenConfig is the input to Open. DatabaseURL is required; everything
// else has reasonable defaults.
type OpenConfig struct {
	DatabaseURL     string
	MaxOpenConns    int
	ConnMaxLifetime time.Duration
	Logger          *slog.Logger
}

// Store owns the Postgres connection pool. It also keeps a *sql.DB
// adapter alive for use by goose and any database/sql-based code; the
// underlying connections come from the same pgxpool so there is no
// double-pool overhead.
type Store struct {
	pool   *pgxpool.Pool
	sqldb  *sql.DB
	logger *slog.Logger
}

// Open initializes the pool and runs any pending migrations. Callers
// own the returned *Store and must call Close on shutdown.
func Open(ctx context.Context, cfg OpenConfig) (*Store, error) {
	if cfg.DatabaseURL == "" {
		return nil, errors.New("storage: DatabaseURL is required")
	}
	logger := cfg.Logger
	if logger == nil {
		logger = slog.Default()
	}

	poolCfg, err := pgxpool.ParseConfig(cfg.DatabaseURL)
	if err != nil {
		return nil, fmt.Errorf("storage: parse database url: %w", err)
	}
	if cfg.MaxOpenConns > 0 {
		// pgxpool wants int32; clamp at the boundary so callers can
		// keep an idiomatic int in their config struct.
		poolCfg.MaxConns = int32(min(cfg.MaxOpenConns, math.MaxInt32)) //nolint:gosec // clamped above
	}
	if cfg.ConnMaxLifetime > 0 {
		poolCfg.MaxConnLifetime = cfg.ConnMaxLifetime
	}

	pool, err := pgxpool.NewWithConfig(ctx, poolCfg)
	if err != nil {
		return nil, fmt.Errorf("storage: open pgxpool: %w", err)
	}

	// Verify connectivity early so callers get a clear error rather than
	// the first lazy query failing later.
	if err := pool.Ping(ctx); err != nil {
		pool.Close()
		return nil, fmt.Errorf("storage: ping: %w", err)
	}

	sqldb := stdlib.OpenDBFromPool(pool)

	if err := runMigrations(ctx, sqldb, logger); err != nil {
		_ = sqldb.Close()
		pool.Close()
		return nil, err
	}

	return &Store{pool: pool, sqldb: sqldb, logger: logger}, nil
}

// Pool returns the pgx connection pool for sqlc-generated queries.
func (s *Store) Pool() *pgxpool.Pool { return s.pool }

// DB returns a *sql.DB adapter sharing the same underlying connections.
// Used by goose and any code preferring database/sql.
func (s *Store) DB() *sql.DB { return s.sqldb }

// Ping reports whether the database is reachable. Used by /readyz.
func (s *Store) Ping(ctx context.Context) error {
	return s.pool.Ping(ctx)
}

// Close releases the connection pool and the *sql.DB adapter.
func (s *Store) Close() {
	if s.sqldb != nil {
		_ = s.sqldb.Close()
	}
	if s.pool != nil {
		s.pool.Close()
	}
}

func runMigrations(ctx context.Context, db *sql.DB, logger *slog.Logger) error {
	sub, err := fs.Sub(migrationsFS, "migrations")
	if err != nil {
		return fmt.Errorf("storage: sub migrations fs: %w", err)
	}
	goose.SetBaseFS(sub)
	if err := goose.SetDialect("postgres"); err != nil {
		return fmt.Errorf("storage: goose dialect: %w", err)
	}
	goose.SetLogger(&gooseSlogAdapter{logger: logger})

	if err := goose.UpContext(ctx, db, "."); err != nil {
		return fmt.Errorf("storage: run migrations: %w", err)
	}
	return nil
}

// gooseSlogAdapter routes goose's logger interface to slog.
type gooseSlogAdapter struct {
	logger *slog.Logger
}

func (g *gooseSlogAdapter) Fatalf(format string, args ...any) {
	g.logger.Error(fmt.Sprintf(format, args...), "source", "goose")
}

func (g *gooseSlogAdapter) Printf(format string, args ...any) {
	g.logger.Info(fmt.Sprintf(format, args...), "source", "goose")
}
