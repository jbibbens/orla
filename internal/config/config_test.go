package config

import (
	"context"
	"testing"
	"time"

	"github.com/sethvargo/go-envconfig"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestLoad_Defaults(t *testing.T) {
	ctx := context.Background()
	lookuper := envconfig.MapLookuper(map[string]string{
		"ORLA_DATABASE_URL": "postgres://localhost/orla",
	})

	cfg, err := loadWith(ctx, lookuper)
	require.NoError(t, err)

	assert.Equal(t, "postgres://localhost/orla", cfg.DatabaseURL)
	assert.Equal(t, "localhost:8081", cfg.ListenAddress)
	assert.Equal(t, "text", cfg.LogFormat)
	assert.Equal(t, "info", cfg.LogLevel)
	assert.Equal(t, int64(10*1024*1024), cfg.MaxRequestBytes)
	assert.Equal(t, 30*time.Second, cfg.ReadTimeout)
	assert.Equal(t, 30*time.Minute, cfg.WriteTimeout)
	assert.Equal(t, 30*time.Second, cfg.ShutdownTimeout)
	assert.Equal(t, 20, cfg.DBMaxOpenConns)
}

func TestLoad_Overrides(t *testing.T) {
	ctx := context.Background()
	lookuper := envconfig.MapLookuper(map[string]string{
		"ORLA_DATABASE_URL":   "postgres://prod/orla",
		"ORLA_LISTEN_ADDRESS": "0.0.0.0:9090",
		"ORLA_LOG_FORMAT":     "json",
		"ORLA_LOG_LEVEL":      "warn",
		"ORLA_WRITE_TIMEOUT":  "1h",
	})

	cfg, err := loadWith(ctx, lookuper)
	require.NoError(t, err)

	assert.Equal(t, "0.0.0.0:9090", cfg.ListenAddress)
	assert.Equal(t, "json", cfg.LogFormat)
	assert.Equal(t, "warn", cfg.LogLevel)
	assert.Equal(t, time.Hour, cfg.WriteTimeout)
}

func TestLoad_MissingDatabaseURL(t *testing.T) {
	ctx := context.Background()
	lookuper := envconfig.MapLookuper(map[string]string{})

	_, err := loadWith(ctx, lookuper)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "DATABASE_URL")
}

func TestLoad_InvalidDuration(t *testing.T) {
	ctx := context.Background()
	lookuper := envconfig.MapLookuper(map[string]string{
		"ORLA_DATABASE_URL":  "postgres://localhost/orla",
		"ORLA_WRITE_TIMEOUT": "not-a-duration",
	})

	_, err := loadWith(ctx, lookuper)
	require.Error(t, err)
}
