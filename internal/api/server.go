// Package api hosts the HTTP server, middleware stack, and request
// handlers. The server uses chi for routing because middleware
// composition (RequestID, Logger, Recoverer, Timeout, BodyLimit) is the
// shape we need and stdlib mux doesn't provide.
package api

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"time"

	"github.com/go-chi/chi/v5"
	"github.com/go-chi/chi/v5/middleware"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

// ReadyFunc reports whether the daemon is fully ready to serve traffic.
// /readyz invokes it with a short timeout; returning a non-nil error
// renders a 503. ReadyFunc may be nil (then /readyz always reports OK).
type ReadyFunc func(ctx context.Context) error

// ServerConfig is the input to NewServer. All durations have sensible
// defaults; the only required field is ListenAddress.
type ServerConfig struct {
	ListenAddress     string
	ReadTimeout       time.Duration
	ReadHeaderTimeout time.Duration
	WriteTimeout      time.Duration
	IdleTimeout       time.Duration

	// MaxRequestBytes caps incoming JSON body size. 0 disables.
	MaxRequestBytes int64

	// Ready is called by /readyz. May be nil.
	Ready ReadyFunc

	// Logger receives request and lifecycle events.
	Logger *slog.Logger

	// PromRegistry, if non-nil, mounts /metrics using a handler that
	// scrapes this registry.
	PromRegistry prometheus.Gatherer
}

// Server wraps an *http.Server and the chi router. Construct with
// NewServer and run with Start/Shutdown.
type Server struct {
	httpServer *http.Server
	router     chi.Router
	logger     *slog.Logger
}

// NewServer assembles the middleware stack and registers the health
// endpoints. Feature routes are added by callers via Router() before
// Start is called.
func NewServer(cfg ServerConfig) *Server {
	if cfg.Logger == nil {
		cfg.Logger = slog.Default()
	}

	r := chi.NewRouter()
	r.Use(middleware.RequestID)
	r.Use(loggingMiddleware(cfg.Logger))
	r.Use(middleware.Recoverer)
	if cfg.MaxRequestBytes > 0 {
		r.Use(bodyLimitMiddleware(cfg.MaxRequestBytes))
	}

	registerHealthRoutes(r, cfg.Ready)
	if cfg.PromRegistry != nil {
		r.Handle("/metrics", promhttp.HandlerFor(cfg.PromRegistry, promhttp.HandlerOpts{}))
	}

	return &Server{
		router: r,
		logger: cfg.Logger,
		httpServer: &http.Server{
			Addr:              cfg.ListenAddress,
			Handler:           r,
			ReadTimeout:       cfg.ReadTimeout,
			ReadHeaderTimeout: cfg.ReadHeaderTimeout,
			WriteTimeout:      cfg.WriteTimeout,
			IdleTimeout:       cfg.IdleTimeout,
		},
	}
}

// Router exposes the underlying chi.Router so feature packages can
// register their routes before Start.
func (s *Server) Router() chi.Router { return s.router }

// Addr returns the resolved listen address (useful in tests that use
// :0 to pick an ephemeral port — only valid after Start).
func (s *Server) Addr() string { return s.httpServer.Addr }

// Start blocks until the HTTP server stops. Returns nil when stopped
// via Shutdown, otherwise the underlying error.
func (s *Server) Start() error {
	s.logger.Info("http server listening", "address", s.httpServer.Addr)
	if err := s.httpServer.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
		return fmt.Errorf("http server: %w", err)
	}
	return nil
}

// Shutdown gracefully drains in-flight requests, bounded by ctx.
func (s *Server) Shutdown(ctx context.Context) error {
	return s.httpServer.Shutdown(ctx)
}

func bodyLimitMiddleware(limit int64) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			r.Body = http.MaxBytesReader(w, r.Body, limit)
			next.ServeHTTP(w, r)
		})
	}
}

func loggingMiddleware(logger *slog.Logger) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			start := time.Now()
			reqID := middleware.GetReqID(r.Context())
			if reqID != "" {
				// Echo to clients so they can quote it in bug reports.
				w.Header().Set("X-Request-Id", reqID)
			}
			ww := middleware.NewWrapResponseWriter(w, r.ProtoMajor)
			next.ServeHTTP(ww, r)

			// Health endpoints are noisy on prod log aggregators.
			level := slog.LevelInfo
			if r.URL.Path == "/healthz" || r.URL.Path == "/readyz" {
				level = slog.LevelDebug
			}
			logger.LogAttrs(r.Context(), level, "http request",
				slog.String("method", r.Method),
				slog.String("path", r.URL.Path),
				slog.Int("status", ww.Status()),
				slog.Int("bytes", ww.BytesWritten()),
				slog.Duration("duration", time.Since(start)),
				slog.String("request_id", reqID),
			)
		})
	}
}
