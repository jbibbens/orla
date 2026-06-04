package api

import (
	"errors"
	"net/http"

	"github.com/go-chi/chi/v5"

	"github.com/harvard-cns/orla/internal/stages"
)

// RegisterStageRoutes mounts the stage endpoints onto r. Routes:
//
//	GET    /api/v1/stages
//	GET    /api/v1/stages/{id}
//	PUT    /api/v1/stages/{id}   (full replace)
//	PATCH  /api/v1/stages/{id}   (partial update)
//	DELETE /api/v1/stages/{id}
//
// Auto-create on first sighting is *not* exposed via REST; it happens
// implicitly inside the proxy when a request arrives for an unknown
// stage. The mapper is expected to use PUT/PATCH for explicit
// configuration.
func RegisterStageRoutes(r chi.Router, reg stages.Registry) {
	h := &stageHandler{reg: reg}
	r.Route("/api/v1/stages", func(r chi.Router) {
		r.Get("/", h.list)
		r.Route("/{id}", func(r chi.Router) {
			r.Get("/", h.get)
			r.Put("/", h.put)
			r.Patch("/", h.patch)
			r.Delete("/", h.delete)
		})
	})
}

type stageHandler struct {
	reg stages.Registry
}

func (h *stageHandler) list(w http.ResponseWriter, r *http.Request) {
	rows, err := h.reg.List(r.Context())
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"stages": rows})
}

func (h *stageHandler) get(w http.ResponseWriter, r *http.Request) {
	id := chi.URLParam(r, "id")
	if id == "" {
		writeErrorMsg(w, http.StatusBadRequest, "id is required")
		return
	}
	s, err := h.reg.Get(r.Context(), id)
	if err != nil {
		if errors.Is(err, stages.ErrNotFound) {
			writeError(w, http.StatusNotFound, err)
			return
		}
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	writeJSON(w, http.StatusOK, s)
}

// putRequest is the wire shape for full-replace. id comes from the URL,
// not the body.
type putRequest struct {
	Backend         string         `json:"backend"`
	ReasoningEffort string         `json:"reasoning_effort"`
	Labels          map[string]any `json:"labels"`
}

func (h *stageHandler) put(w http.ResponseWriter, r *http.Request) {
	id := chi.URLParam(r, "id")
	if id == "" {
		writeErrorMsg(w, http.StatusBadRequest, "id is required")
		return
	}
	var body putRequest
	if !decodeJSON(w, r, &body) {
		return
	}
	s, err := h.reg.Replace(r.Context(), &stages.Stage{
		ID:              id,
		Backend:         body.Backend,
		ReasoningEffort: body.ReasoningEffort,
		Labels:          body.Labels,
	})
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	writeJSON(w, http.StatusOK, s)
}

func (h *stageHandler) patch(w http.ResponseWriter, r *http.Request) {
	id := chi.URLParam(r, "id")
	if id == "" {
		writeErrorMsg(w, http.StatusBadRequest, "id is required")
		return
	}
	var body stages.PatchRequest
	if !decodeJSON(w, r, &body) {
		return
	}
	s, err := h.reg.Patch(r.Context(), id, body)
	if err != nil {
		if errors.Is(err, stages.ErrNotFound) {
			writeError(w, http.StatusNotFound, err)
			return
		}
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	writeJSON(w, http.StatusOK, s)
}

func (h *stageHandler) delete(w http.ResponseWriter, r *http.Request) {
	id := chi.URLParam(r, "id")
	if id == "" {
		writeErrorMsg(w, http.StatusBadRequest, "id is required")
		return
	}
	if err := h.reg.Delete(r.Context(), id); err != nil {
		if errors.Is(err, stages.ErrNotFound) {
			writeError(w, http.StatusNotFound, err)
			return
		}
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	w.WriteHeader(http.StatusNoContent)
}
