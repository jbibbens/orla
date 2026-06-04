// Package structurepred implements the ToolProvider interface for
// protein-structure-prediction backends (Boltz-2, Chai-1, Protenix).
//
// The wire format is intentionally minimal: a POST to
// {endpoint}/v1/tools/structure-prediction with a JSON body matching
// Request and a JSON response matching Response. The body is wrapped
// in provider.ToolRequest/ToolResponse for transport over the generic
// /v1/tools/{kind} orla route.
//
// All three target models (Boltz-2, Chai-1, Protenix) are MSA-free and
// take FASTA + optional ligand SMILES, return CIF + per-residue
// confidence scores. The wrapper service on each backend translates
// this schema to the tool-specific API.
package structurepred

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"time"

	"github.com/harvard-cns/orla/internal/backends"
	"github.com/harvard-cns/orla/internal/provider"
)

// ToolKind is the canonical kind string for this provider.
const ToolKind = "structure-prediction"

// Request is the wire shape POSTed to a structure-prediction wrapper.
type Request struct {
	// Sequences is the list of protein FASTA sequences. For a monomer
	// prediction supply one entry; for a complex supply multiple.
	Sequences []string `json:"sequences"`

	// LigandSMILES is the optional ligand SMILES — supply one entry
	// per ligand to co-fold. Empty list means protein-only prediction.
	LigandSMILES []string `json:"ligand_smiles,omitempty"`

	// Options is a kind-of-tool-specific opaque map (e.g.,
	// {"num_recycles": 3, "seed": 42}). The wrapper service interprets
	// it per backend; orla passes it through verbatim.
	Options map[string]any `json:"options,omitempty"`
}

// Response is the wire shape returned by the wrapper.
type Response struct {
	// StructureCIF is the predicted structure in CIF format. May be
	// large (~100 KB to a few MB).
	StructureCIF string `json:"structure_cif"`

	// PLDDTPerResidue is the per-residue predicted-LDDT confidence
	// (0-100), one entry per residue, concatenated across sequences.
	PLDDTPerResidue []float64 `json:"plddt_per_residue,omitempty"`

	// PTMScore is the predicted TM-score (0-1). NULL when not reported
	// by the underlying tool.
	PTMScore *float64 `json:"ptm_score,omitempty"`

	// IPTMScore is the interface predicted TM-score for complexes.
	// NULL for monomers or when not reported.
	IPTMScore *float64 `json:"iptm_score,omitempty"`
}

// Client implements provider.ToolProvider against a single
// HTTP-hosted structure-prediction wrapper.
type Client struct {
	name     string
	endpoint string
	apiKey   string
	httpc    *http.Client
}

// Compile-time interface check.
var _ provider.ToolProvider = (*Client)(nil)

// New builds a Client from a backend record. The API key (if any) is
// resolved via os.Getenv(b.APIKeyEnvVar); an unset value yields an
// anonymous client. Endpoint must already include the scheme (http://
// or https://); it is treated as a base URL.
func New(b *backends.Backend) *Client {
	c := &Client{
		name:     b.Name,
		endpoint: b.Endpoint,
		// Structure-prediction calls can take minutes — generous timeout.
		// The orla-side context still bounds the call separately.
		httpc: &http.Client{Timeout: 10 * time.Minute},
	}
	if b.APIKeyEnvVar != "" {
		c.apiKey = os.Getenv(b.APIKeyEnvVar)
	}
	return c
}

func (c *Client) Name() string     { return c.name }
func (c *Client) ToolKind() string { return ToolKind }

// Invoke decodes the inner Request from the envelope, POSTs to the
// remote wrapper, decodes the Response, and re-wraps it for return.
//
// Network and HTTP errors are returned as-is wrapped with backend
// context. Non-200 statuses are decoded as the OpenAI-shape error
// envelope when possible, plain text otherwise.
func (c *Client) Invoke(ctx context.Context, req provider.ToolRequest) (*provider.ToolResponse, error) {
	if req.Kind != "" && req.Kind != ToolKind {
		return nil, fmt.Errorf("structurepred: wrong kind %q (want %q)", req.Kind, ToolKind)
	}
	body, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("structurepred: encode request: %w", err)
	}

	url := c.endpoint + "/v1/tools/" + ToolKind
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("structurepred: build request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	if c.apiKey != "" {
		httpReq.Header.Set("Authorization", "Bearer "+c.apiKey)
	}

	resp, err := c.httpc.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("structurepred: POST: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	rawBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("structurepred: read response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("structurepred: backend %s: status=%d body=%s",
			c.name, resp.StatusCode, truncate(rawBody, 500))
	}

	var out provider.ToolResponse
	if err := json.Unmarshal(rawBody, &out); err != nil {
		return nil, fmt.Errorf("structurepred: decode response: %w", err)
	}
	return &out, nil
}

func truncate(b []byte, n int) string {
	if len(b) <= n {
		return string(b)
	}
	return string(b[:n]) + "..."
}
