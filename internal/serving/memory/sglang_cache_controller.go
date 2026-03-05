package memory

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/dorcha-inc/orla/internal/core"
)

const (
	sglangHTTPTimeout = 5 * time.Second
	sglangPathFlush   = "/flush_cache"
)

// SGLangCacheController implements CacheController for SGLang backends.
// It uses SGLang's /flush_cache and /get_server_info HTTP endpoints.
type SGLangCacheController struct {
	baseURL string
	client  *http.Client
}

// NewSGLangCacheController creates a CacheController for an SGLang backend.
// baseURL is the root URL of the SGLang server (e.g. "http://sglang:30000").
func NewSGLangCacheController(baseURL string) *SGLangCacheController {
	return &SGLangCacheController{
		baseURL: strings.TrimRight(baseURL, "/"),
		client:  &http.Client{Timeout: sglangHTTPTimeout},
	}
}

// FlushPrefix triggers a global KV cache flush on the SGLang backend.
// SGLang's /flush_cache is a global operation; the sessionID parameter is
// logged but not used for scoping because SGLang does not support per-session eviction.
func (c *SGLangCacheController) FlushPrefix(ctx context.Context, _ string) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+sglangPathFlush, nil)
	if err != nil {
		return fmt.Errorf("sglang flush: build request: %w", err)
	}
	resp, err := c.client.Do(req)
	if err != nil {
		return fmt.Errorf("sglang flush: %w", err)
	}
	defer core.LogDeferredError(resp.Body.Close)
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("sglang flush: unexpected status %d", resp.StatusCode)
	}
	return nil
}

// sglangServerInfo is the subset of fields we care about from /get_server_info.
type sglangServerInfo struct {
	MaxTotalNumTokens int64   `json:"max_total_num_tokens"`
	KVCacheUsage      float64 `json:"kv_cache_usage"` // 0.0 to 1.0 if available
}

// MemoryUsage queries the SGLang backend for KV cache utilization.
func (c *SGLangCacheController) MemoryUsage(ctx context.Context) (*MemoryStats, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.baseURL+"/get_server_info", nil)
	if err != nil {
		return nil, fmt.Errorf("sglang memory: build request: %w", err)
	}
	resp, err := c.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("sglang memory: %w", err)
	}
	defer core.LogDeferredError(resp.Body.Close)
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("sglang memory: unexpected status %d", resp.StatusCode)
	}

	var info sglangServerInfo
	if err := json.NewDecoder(resp.Body).Decode(&info); err != nil {
		return nil, fmt.Errorf("sglang memory: decode: %w", err)
	}

	pressure := info.KVCacheUsage
	var usedBytes, totalBytes int64
	if info.MaxTotalNumTokens > 0 {
		totalBytes = info.MaxTotalNumTokens
		usedBytes = int64(float64(totalBytes) * pressure)
	}

	return &MemoryStats{
		UsedBytes:  usedBytes,
		TotalBytes: totalBytes,
		Pressure:   pressure,
	}, nil
}
