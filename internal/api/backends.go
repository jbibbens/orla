package api

import (
	"errors"
	"net/http"

	"github.com/go-chi/chi/v5"

	"github.com/harvard-cns/orla/internal/backends"
)

// BackendLifecycle is the subset of scheduler.Scheduler the backend
// handlers call to keep the runtime in sync with the registry.
// Optional — pass nil in tests that don't exercise dispatch.
type BackendLifecycle interface {
	Register(b *backends.Backend)
	Deregister(name string)
}

// BackendDeps bundles backend-handler dependencies.
type BackendDeps struct {
	Registry  backends.Registry
	Lifecycle BackendLifecycle
}

// RegisterBackendRoutes mounts the backend endpoints onto r. Routes:
//
//	POST   /api/v1/backends         (create; name in body)
//	GET    /api/v1/backends         (list)
//	GET    /api/v1/backends/{name}
//	PATCH  /api/v1/backends/{name}
//	DELETE /api/v1/backends/{name}
//
// There is no PUT; backends are explicitly created with POST. To change
// immutable fields (name, model_id), delete and re-create.
func RegisterBackendRoutes(r chi.Router, deps BackendDeps) {
	h := &backendHandler{deps: deps}
	r.Route("/api/v1/backends", func(r chi.Router) {
		r.Post("/", h.create)
		r.Get("/", h.list)
		r.Route("/{name}", func(r chi.Router) {
			r.Get("/", h.get)
			r.Patch("/", h.patch)
			r.Delete("/", h.delete)
		})
	})
}

type backendHandler struct {
	deps BackendDeps
}

// createRequest is the POST wire shape. name lives in the body for
// symmetry with how the daemon's own RegisterBackend invocations look;
// otherwise we'd need a sub-resource collection trick.
//
// Kind defaults to "llm" when omitted. For "llm", ModelID is required;
// for "tool", ToolKind is required.
type createRequest struct {
	Name                string   `json:"name"`
	Kind                string   `json:"kind,omitempty"`
	Endpoint            string   `json:"endpoint"`
	APIKeyEnvVar        string   `json:"api_key_env_var"`
	MaxConcurrency      int32    `json:"max_concurrency"`
	Quality             *float64 `json:"quality"`
	RatePerSecond       *float64 `json:"rate_per_second"`

	// LLM fields:
	ModelID             string   `json:"model_id,omitempty"`
	InputCostPerMtoken  *float64 `json:"input_cost_per_mtoken,omitempty"`
	OutputCostPerMtoken *float64 `json:"output_cost_per_mtoken,omitempty"`

	// Tool fields:
	ToolKind         string   `json:"tool_kind,omitempty"`
	CostPerGPUSecond *float64 `json:"cost_per_gpu_second,omitempty"`
}

func (h *backendHandler) create(w http.ResponseWriter, r *http.Request) {
	var req createRequest
	if !decodeJSON(w, r, &req) {
		return
	}
	if req.Name == "" {
		writeErrorMsg(w, http.StatusBadRequest, "name is required")
		return
	}
	if req.Endpoint == "" {
		writeErrorMsg(w, http.StatusBadRequest, "endpoint is required")
		return
	}
	if req.MaxConcurrency < 1 {
		writeErrorMsg(w, http.StatusBadRequest, "max_concurrency must be >= 1")
		return
	}
	if req.RatePerSecond != nil && *req.RatePerSecond < 0 {
		writeErrorMsg(w, http.StatusBadRequest, "rate_per_second must be >= 0")
		return
	}

	kind := backends.Kind(req.Kind)
	if kind == "" {
		kind = backends.KindLLM
	}
	switch kind {
	case backends.KindLLM:
		if req.ModelID == "" {
			writeErrorMsg(w, http.StatusBadRequest, "model_id is required for kind=llm")
			return
		}
	case backends.KindTool:
		if req.ToolKind == "" {
			writeErrorMsg(w, http.StatusBadRequest, "tool_kind is required for kind=tool")
			return
		}
	default:
		writeErrorMsg(w, http.StatusBadRequest,
			"kind must be 'llm' or 'tool'")
		return
	}

	b := &backends.Backend{
		Name:                req.Name,
		Endpoint:             req.Endpoint,
		APIKeyEnvVar:         req.APIKeyEnvVar,
		MaxConcurrency:       req.MaxConcurrency,
		InputCostPerMtoken:   req.InputCostPerMtoken,
		OutputCostPerMtoken:  req.OutputCostPerMtoken,
		Quality:              req.Quality,
		RatePerSecond:        req.RatePerSecond,
		Kind:                 kind,
		CostPerGPUSecond:     req.CostPerGPUSecond,
	}
	if req.ModelID != "" {
		b.ModelID = &req.ModelID
	}
	if req.ToolKind != "" {
		b.ToolKind = &req.ToolKind
	}
	b, err := h.deps.Registry.Insert(r.Context(), b)
	if err != nil {
		if errors.Is(err, backends.ErrAlreadyExists) {
			writeError(w, http.StatusConflict, err)
			return
		}
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	if h.deps.Lifecycle != nil {
		h.deps.Lifecycle.Register(b)
	}
	writeJSON(w, http.StatusCreated, b)
}

func (h *backendHandler) list(w http.ResponseWriter, r *http.Request) {
	rows, err := h.deps.Registry.List(r.Context())
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"backends": rows})
}

func (h *backendHandler) get(w http.ResponseWriter, r *http.Request) {
	name := chi.URLParam(r, "name")
	b, err := h.deps.Registry.Get(r.Context(), name)
	if err != nil {
		if errors.Is(err, backends.ErrNotFound) {
			writeError(w, http.StatusNotFound, err)
			return
		}
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	writeJSON(w, http.StatusOK, b)
}

func (h *backendHandler) patch(w http.ResponseWriter, r *http.Request) {
	name := chi.URLParam(r, "name")
	var req backends.PatchRequest
	if !decodeJSON(w, r, &req) {
		return
	}
	if req.MaxConcurrency != nil && *req.MaxConcurrency < 1 {
		writeErrorMsg(w, http.StatusBadRequest, "max_concurrency must be >= 1")
		return
	}
	b, err := h.deps.Registry.Patch(r.Context(), name, req)
	if err != nil {
		if errors.Is(err, backends.ErrNotFound) {
			writeError(w, http.StatusNotFound, err)
			return
		}
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	// Re-register so the scheduler picks up changes to MaxConcurrency,
	// endpoint, or API key env var.
	if h.deps.Lifecycle != nil {
		h.deps.Lifecycle.Register(b)
	}
	writeJSON(w, http.StatusOK, b)
}

func (h *backendHandler) delete(w http.ResponseWriter, r *http.Request) {
	name := chi.URLParam(r, "name")
	if err := h.deps.Registry.Delete(r.Context(), name); err != nil {
		if errors.Is(err, backends.ErrNotFound) {
			writeError(w, http.StatusNotFound, err)
			return
		}
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	if h.deps.Lifecycle != nil {
		h.deps.Lifecycle.Deregister(name)
	}
	w.WriteHeader(http.StatusNoContent)
}
