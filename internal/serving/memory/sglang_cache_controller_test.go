package memory

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestSGLangCacheController_FlushPrefix_Success(t *testing.T) {
	var called atomic.Bool
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == sglangPathFlush && r.Method == http.MethodPost {
			called.Store(true)
			w.WriteHeader(http.StatusOK)
			return
		}
		http.NotFound(w, r)
	}))
	defer srv.Close()

	cc := NewSGLangCacheController(srv.URL)
	err := cc.FlushPrefix(context.Background(), "wf1")
	require.NoError(t, err)
	assert.True(t, called.Load())
}

func TestSGLangCacheController_FlushPrefix_ServerError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer srv.Close()

	cc := NewSGLangCacheController(srv.URL)
	err := cc.FlushPrefix(context.Background(), "wf1")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "unexpected status 500")
}

func TestSGLangCacheController_FlushPrefix_Unreachable(t *testing.T) {
	cc := NewSGLangCacheController("http://127.0.0.1:1")
	err := cc.FlushPrefix(context.Background(), "wf1")
	require.Error(t, err)
}

func TestSGLangCacheController_MemoryUsage_Success(t *testing.T) {
	info := map[string]any{
		"max_total_num_tokens": 8192,
		"kv_cache_usage":       0.42,
	}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/get_server_info" && r.Method == http.MethodGet {
			w.Header().Set("Content-Type", "application/json")
			require.NoError(t, json.NewEncoder(w).Encode(info))
			return
		}
		http.NotFound(w, r)
	}))
	defer srv.Close()

	cc := NewSGLangCacheController(srv.URL)
	stats, err := cc.MemoryUsage(context.Background())
	require.NoError(t, err)
	assert.InDelta(t, 0.42, stats.Pressure, 0.001)
	assert.Equal(t, int64(8192), stats.TotalBytes)
	assert.Equal(t, int64(3440), stats.UsedBytes) // 8192 * 0.42 ≈ 3440
}

func TestSGLangCacheController_MemoryUsage_ServerError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusServiceUnavailable)
	}))
	defer srv.Close()

	cc := NewSGLangCacheController(srv.URL)
	_, err := cc.MemoryUsage(context.Background())
	require.Error(t, err)
	assert.Contains(t, err.Error(), "unexpected status 503")
}

func TestSGLangCacheController_MemoryUsage_BadJSON(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, err := w.Write([]byte("not json"))
		require.NoError(t, err)
	}))
	defer srv.Close()

	cc := NewSGLangCacheController(srv.URL)
	_, err := cc.MemoryUsage(context.Background())
	require.Error(t, err)
	assert.Contains(t, err.Error(), "decode")
}

func TestDefaultManager_FlushCallsCacheController(t *testing.T) {
	var flushed atomic.Bool
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == sglangPathFlush {
			flushed.Store(true)
			w.WriteHeader(http.StatusOK)
			return
		}
		http.NotFound(w, r)
	}))
	defer srv.Close()

	mm := NewDefaultManager(DefaultManagerConfig{})
	cc := NewSGLangCacheController(srv.URL)
	mm.RegisterCacheController("b1", cc)
	mm.RegisterWorkflow("wf1")

	action := mm.OnTransition(context.Background(), StageTransition{
		TransitionType: TransitionWorkflowComplete,
		WorkflowID:     "wf1",
		Backend:        "b1",
	})
	assert.Equal(t, CacheActionFlush, action.Type)
	assert.True(t, flushed.Load(), "SGLang /flush_cache should have been called")
}

func TestDefaultManager_FlushSkippedWhenOtherInflight(t *testing.T) {
	var flushed atomic.Bool
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == sglangPathFlush {
			flushed.Store(true)
			w.WriteHeader(http.StatusOK)
			return
		}
		http.NotFound(w, r)
	}))
	defer srv.Close()

	mm := NewDefaultManager(DefaultManagerConfig{})
	cc := NewSGLangCacheController(srv.URL)
	mm.RegisterCacheController("b1", cc)
	mm.RegisterWorkflow("wf1")
	mm.RegisterWorkflow("wf2")

	mm.RecordInflight(InflightRequest{
		RequestID: "r2", WorkflowID: "wf2", StageID: "s2", Backend: "b1",
	})

	action := mm.OnTransition(context.Background(), StageTransition{
		TransitionType: TransitionWorkflowComplete,
		WorkflowID:     "wf1",
		Backend:        "b1",
	})
	assert.Equal(t, CacheActionFlush, action.Type)
	assert.False(t, flushed.Load(), "should NOT call /flush_cache when another workflow is in-flight")
}
