package storage

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/testcontainers/testcontainers-go"
	"github.com/testcontainers/testcontainers-go/modules/postgres"
	"github.com/testcontainers/testcontainers-go/wait"
)

// pgContainer brings up a Postgres container and returns its DSN. The
// container is automatically cleaned up via t.Cleanup.
func pgContainer(t *testing.T) string {
	t.Helper()
	ctx := context.Background()

	pgC, err := postgres.Run(ctx,
		"postgres:17",
		postgres.WithDatabase("orla"),
		postgres.WithUsername("orla"),
		postgres.WithPassword("orla"),
		testcontainers.WithWaitStrategy(
			wait.ForLog("database system is ready to accept connections").
				WithOccurrence(2).
				WithStartupTimeout(60*time.Second),
		),
	)
	require.NoError(t, err, "start postgres container")

	t.Cleanup(func() {
		if err := pgC.Terminate(context.Background()); err != nil {
			t.Logf("terminate postgres container: %v", err)
		}
	})

	dsn, err := pgC.ConnectionString(ctx, "sslmode=disable")
	require.NoError(t, err)
	return dsn
}

func TestOpen_RunsMigrationsAndPings(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping storage test in -short mode")
	}
	dsn := pgContainer(t)

	store, err := Open(context.Background(), OpenConfig{
		DatabaseURL:  dsn,
		MaxOpenConns: 4,
	})
	require.NoError(t, err)
	defer store.Close()

	assert.NoError(t, store.Ping(context.Background()))

	// schema_migrations row should exist (goose ran 0001_init).
	var count int
	row := store.DB().QueryRowContext(context.Background(),
		`SELECT COUNT(*) FROM goose_db_version WHERE version_id > 0`)
	require.NoError(t, row.Scan(&count))
	assert.GreaterOrEqual(t, count, 1, "expected at least one migration applied")
}

func TestOpen_MissingDatabaseURL(t *testing.T) {
	_, err := Open(context.Background(), OpenConfig{})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "DatabaseURL")
}

func TestOpen_InvalidDatabaseURL(t *testing.T) {
	_, err := Open(context.Background(), OpenConfig{
		DatabaseURL: "not a valid url",
	})
	require.Error(t, err)
}

func TestStore_CloseIdempotency(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping storage test in -short mode")
	}
	dsn := pgContainer(t)

	store, err := Open(context.Background(), OpenConfig{DatabaseURL: dsn})
	require.NoError(t, err)
	store.Close()
	// Second close should not panic. (database/sql.DB.Close returns nil
	// after first close; pgxpool.Pool.Close is idempotent.)
	assert.NotPanics(t, store.Close)
}
