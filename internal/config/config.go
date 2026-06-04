// Package config loads the orla daemon configuration from environment
// variables (prefixed with ORLA_).
//
// Every option has a sensible default except DATABASE_URL, which is
// required. There is no YAML config file in v1; env vars are the single
// source of truth so containerized deployments work without a sidecar
// config file.
package config

import (
	"context"
	"time"

	"github.com/sethvargo/go-envconfig"
)

// EnvPrefix is the prefix applied to every env var bound to Config.
const EnvPrefix = "ORLA_"

// Config is the full daemon configuration. Defaults live in the struct
// tags; precedence is env var > default.
type Config struct {
	// DatabaseURL is the Postgres connection URL. Required.
	DatabaseURL string `env:"DATABASE_URL, required"`

	// ListenAddress is the address the HTTP server binds to.
	ListenAddress string `env:"LISTEN_ADDRESS, default=localhost:8081"`

	// LogFormat is "text" or "json".
	LogFormat string `env:"LOG_FORMAT, default=text"`

	// LogLevel is "debug", "info", "warn", or "error".
	LogLevel string `env:"LOG_LEVEL, default=info"`

	// MaxRequestBytes is the maximum body size for non-streaming JSON
	// endpoints. Streaming endpoints are unconstrained.
	MaxRequestBytes int64 `env:"MAX_REQUEST_BYTES, default=10485760"` // 10 MB

	// HTTP server timeouts. WriteTimeout is long because streaming
	// chat completions can run for minutes.
	ReadTimeout       time.Duration `env:"READ_TIMEOUT, default=30s"`
	ReadHeaderTimeout time.Duration `env:"READ_HEADER_TIMEOUT, default=10s"`
	WriteTimeout      time.Duration `env:"WRITE_TIMEOUT, default=30m"`
	IdleTimeout       time.Duration `env:"IDLE_TIMEOUT, default=120s"`

	// ShutdownTimeout is the maximum time graceful shutdown is allowed
	// before the process exits.
	ShutdownTimeout time.Duration `env:"SHUTDOWN_TIMEOUT, default=30s"`

	// Postgres pool settings.
	DBMaxOpenConns    int           `env:"DB_MAX_OPEN_CONNS, default=20"`
	DBMaxIdleConns    int           `env:"DB_MAX_IDLE_CONNS, default=20"`
	DBConnMaxLifetime time.Duration `env:"DB_CONN_MAX_LIFETIME, default=30m"`
}

// Load reads Config from process environment variables. Validation
// errors (e.g. missing DATABASE_URL) are returned as-is from envconfig.
func Load(ctx context.Context) (*Config, error) {
	return loadWith(ctx, envconfig.OsLookuper())
}

// loadWith is the test seam for injecting a custom lookuper.
func loadWith(ctx context.Context, lookuper envconfig.Lookuper) (*Config, error) {
	var c Config
	if err := envconfig.ProcessWith(ctx, &envconfig.Config{
		Target:   &c,
		Lookuper: envconfig.PrefixLookuper(EnvPrefix, lookuper),
	}); err != nil {
		return nil, err
	}
	return &c, nil
}
