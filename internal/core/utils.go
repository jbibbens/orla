package core

import (
	"encoding/json"
	"fmt"
	"maps"
	"net/http"
	"slices"
	"strings"

	"go.uber.org/zap"
)

// Ptr returns a pointer to v. Useful for converting literals to pointers in struct initializers.
func Ptr[T any](v T) *T { return &v }

// JoinMapKeys joins the keys of a map into a comma-separated string.
// Useful for error messages that need to list valid values.
func JoinMapKeys[T comparable](m map[T]struct{}) string {
	keys := slices.Collect(maps.Keys(m))
	sliceStrings := make([]string, len(keys))
	for i, k := range keys {
		sliceStrings[i] = fmt.Sprintf("%v", k)
	}
	return strings.Join(sliceStrings, ", ")
}

// WriteJSONResponse encodes v as JSON and writes it to w. Sets Content-Type: application/json.
func WriteJSONResponse(w http.ResponseWriter, v any) {
	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(v); err != nil {
		zap.L().Error("failed to encode JSON response", zap.Error(err))
		http.Error(w, "Failed to encode JSON response", http.StatusInternalServerError)
	}
}

// WriteSSEResponse writes a single SSE event: "event: <name>\ndata: <json>\n\n".
// On error it only logs (http.Error would be invalid mid-stream). It also flushes
// the response if flusher is not nil.
func WriteSSEResponse(w http.ResponseWriter, flusher http.Flusher, event string, data any) {
	payload, err := json.Marshal(data)
	if err != nil {
		zap.L().Error("failed to marshal SSE event", zap.Error(err))
		return
	}
	if _, err := fmt.Fprintf(w, "event: %s\ndata: %s\n\n", event, payload); err != nil {
		zap.L().Error("failed to write SSE event", zap.Error(err))
		return
	}
	if flusher != nil {
		flusher.Flush()
	}
}
