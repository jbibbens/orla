package api

import (
	"context"
	"encoding/json"
	"net/http"
	"time"

	"github.com/go-chi/chi/v5"
)

const readyTimeout = 2 * time.Second

func registerHealthRoutes(r chi.Router, ready ReadyFunc) {
	r.Get("/healthz", healthzHandler())
	r.Get("/readyz", readyzHandler(ready))
}

// healthzHandler reports process liveness. It always returns 200 as
// long as the process is running; load balancers use this to decide
// whether to keep the pod in service.
func healthzHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, _ *http.Request) {
		writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
	}
}

// readyzHandler reports readiness to serve real traffic. It runs the
// supplied ReadyFunc with a short timeout; failures render a 503.
func readyzHandler(ready ReadyFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if ready == nil {
			writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
			return
		}
		ctx, cancel := context.WithTimeout(r.Context(), readyTimeout)
		defer cancel()
		if err := ready(ctx); err != nil {
			writeJSON(w, http.StatusServiceUnavailable, map[string]string{
				"status": "not_ready",
				"error":  err.Error(),
			})
			return
		}
		writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
	}
}

func writeJSON(w http.ResponseWriter, status int, body any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(body)
}
