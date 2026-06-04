# orla extension plan — tool backends + PoseBusters demo

This is the architectural diff for adding a second backend kind to
orla (`tool`, alongside the existing `llm`) so it can dispatch and
route scientific computation tools — structure prediction, docking,
property prediction — in addition to OpenAI-compatible chat models.

The first concrete tool kind is **structure prediction**. Three
backends: Boltz-2, Chai-1, Protenix. Evaluated on PoseBusters. Hosted
on Nebius GPU instances.

The repo has no external API users; the diff prioritizes the cleanest
end state over migration-friendliness.

## 1. The shape of the abstraction

### Today

A backend is implicitly an OpenAI-compatible chat completion endpoint:

```go
// internal/provider/provider.go
type Provider interface {
    Name() string
    ModelID() string
    Chat(ctx, openai.ChatCompletionNewParams) (*openai.ChatCompletion, error)
    ChatStream(ctx, openai.ChatCompletionNewParams) *ssestream.Stream[openai.ChatCompletionChunk]
}
```

The scheduler, registry, telemetry, mapper, and proxy all assume this
shape. Cost is token-denominated. Routes are
`POST /v1/chat/completions`.

### After

Two backend kinds: `llm` (today's shape) and `tool` (new). A backend
declares its kind at registration time. The scheduler is kind-agnostic
(it manages slots, rate limits, accounting); the proxy is kind-aware
(separate routes per kind, different request/response schemas).

```go
// internal/provider/provider.go — common to all backend kinds
type Backend interface {
    Name() string
    Kind() Kind  // KindLLM | KindTool
}

type Kind string
const (
    KindLLM  Kind = "llm"
    KindTool Kind = "tool"
)

// internal/provider/llm.go — rename from provider.go's Provider
type LLMProvider interface {
    Backend
    ModelID() string
    Chat(ctx, openai.ChatCompletionNewParams) (*openai.ChatCompletion, error)
    ChatStream(ctx, openai.ChatCompletionNewParams) *ssestream.Stream[openai.ChatCompletionChunk]
}

// internal/provider/tool.go — new
type ToolProvider interface {
    Backend
    ToolKind() string  // e.g. "structure-prediction"
    Invoke(ctx, ToolRequest) (*ToolResponse, error)
}

type ToolRequest struct {
    Kind    string          // matches ToolProvider.ToolKind()
    Payload json.RawMessage // kind-specific shape
}

type ToolResponse struct {
    Payload    json.RawMessage
    GPUSeconds float64 // populated by the wrapper for cost accounting
    Metadata   map[string]any
}
```

For the first concrete tool kind, we define Go structs for the
payload (in a sub-package `internal/provider/structurepred/`):

```go
type Request struct {
    Sequences    []string `json:"sequences"`              // FASTA
    LigandSMILES []string `json:"ligand_smiles,omitempty"`
    Options      map[string]any `json:"options,omitempty"`
}

type Response struct {
    StructureCIF    string    `json:"structure_cif"`
    PLDDTPerResidue []float64 `json:"plddt_per_residue,omitempty"`
    PTMScore        *float64  `json:"ptm_score,omitempty"`
    IPTMScore       *float64  `json:"iptm_score,omitempty"`
}
```

## 2. File-level diff

### `internal/provider/`

- `provider.go` → split into `provider.go` (defines `Backend`, `Kind`),
  `llm.go` (existing `Provider` renamed to `LLMProvider`), `tool.go`
  (new `ToolProvider` + `ToolRequest`/`ToolResponse`)
- `openai.go` → existing OpenAI provider implements `LLMProvider`
- `mock.go` → update to satisfy new interfaces; add a `MockToolProvider`
- `structurepred/client.go` → new. HTTP client that POSTs
  `StructurePredictionRequest` to a Nebius-hosted endpoint and reads
  back the response. Implements `ToolProvider`.

### `internal/backends/`

- `backend.go` → `Backend` struct gains `Kind string`, removes
  `ModelID` requirement (only LLM kinds need it), adds
  `CostPerGPUSecond *float64`, adds `ToolKind *string` (e.g.,
  `"structure-prediction"`)
- The cost model branches on Kind:
  - LLM: token-based (unchanged)
  - tool: `gpu_seconds * cost_per_gpu_second`
- Registry validates Kind on insert.
- Test fixtures (`registry_test.go`) get an `InsertTool` case.

### `internal/storage/migrations/`

Add `0007_backend_kinds.sql`:

```sql
-- +goose Up
ALTER TABLE backends
    ADD COLUMN kind TEXT NOT NULL DEFAULT 'llm',
    ADD COLUMN tool_kind TEXT,
    ADD COLUMN cost_per_gpu_second DOUBLE PRECISION;

-- For tool backends, model_id/input_cost/output_cost are unused but kept
-- nullable for schema simplicity. The proxy + registry enforce
-- kind-appropriate fields at the application layer.
ALTER TABLE backends
    ALTER COLUMN model_id DROP NOT NULL;

-- +goose Down
ALTER TABLE backends
    DROP COLUMN cost_per_gpu_second,
    DROP COLUMN tool_kind,
    DROP COLUMN kind;
ALTER TABLE backends
    ALTER COLUMN model_id SET NOT NULL;
```

Add `0008_completion_records_tool_fields.sql`:

```sql
-- +goose Up
ALTER TABLE completion_records
    ADD COLUMN gpu_seconds DOUBLE PRECISION,
    ADD COLUMN tool_kind TEXT;
-- Existing prompt_tokens / completion_tokens stay; they're NULL for
-- tool dispatches. gpu_seconds is NULL for LLM dispatches.

-- +goose Down
ALTER TABLE completion_records
    DROP COLUMN tool_kind,
    DROP COLUMN gpu_seconds;
```

### `internal/storage/queries/`

- `backends.sql` updated to include the new columns in select/insert/
  update queries.
- `completion_records.sql` updated to include `gpu_seconds`, `tool_kind`
  in inserts and metric rollups (`StageMetricsByBackend` returns per-
  backend cost regardless of kind because cost_usd is the canonical
  derived field).

### `internal/scheduler/scheduler.go`

The scheduler is **already** kind-agnostic in its core machinery —
slots, rate limits, dispatch counters work the same regardless of
what the Provider type is doing under the hood. Concrete changes:

- `ProviderFactory` returns the new `Backend` interface (not the
  current `provider.Provider`).
- `Scheduler.Provider(name)` → renamed to `Scheduler.Backend(name)`,
  returns `Backend`. Callers cast to `LLMProvider` or `ToolProvider`.
- `Scheduler.Dispatch` is removed (it was sugar for LLM only). Proxy
  layer keeps its own dispatch logic per kind.
- Tests get fixture tool providers; concurrency/rate-limit tests are
  unchanged (kind-agnostic).

### `internal/api/`

- `proxy.go` → keep `chatCompletions` handler exactly as-is at
  `POST /v1/chat/completions`. It validates that the resolved
  backend's Kind is `llm`; returns 400 if not.
- `tool.go` → new. Handler at `POST /v1/tools/{kind}`. Decodes
  payload by `kind`, validates backend Kind matches, calls
  `ToolProvider.Invoke`, returns response. Records completion +
  cost. Posts feedback rows the same way the chat path does.
- `server.go` → register both routes.

### `internal/telemetry/`

- `completion.go` → `CompletionRecord` gains `GPUSeconds *float64`
  and `ToolKind string`. Flush function copies both fields.
- `reader.go` → unchanged.

### `internal/mapper/` (in demo, not orla core)

No changes. The mapper reads (rating, cost, latency) per (stage,
backend) and that contract is identical for both kinds.

### `cmd/orla/commands/serve.go`

Update the scheduler factory to dispatch to the right concrete
provider type based on Kind:

```go
sched := scheduler.New(func(b *backends.Backend) provider.Backend {
    switch b.Kind {
    case "tool":
        return structurepred.NewClient(b)
    default:
        return provider.NewOpenAI(b)
    }
}, logger)
```

## 3. The Nebius infrastructure

**One Nebius Managed Kubernetes cluster** with 3 GPU nodes (one A100-80GB
per tool). Each tool is a standard Kubernetes Deployment + Service —
no Ray, no KubeRay, no Helm chart soup. Each Deployment runs:

1. The tool itself (Boltz-2, Chai-1, or Protenix) — installed via the
   project's documented method (HF weights, conda/pip), baked into a
   container image.
2. A small FastAPI server (~150 LOC each) exposing:
   - `POST /v1/tools/structure-prediction`
   - Bearer-token auth (`ORLA_TOOL_TOKEN` env var)
   - JSON request matching `structurepred.Request`
   - JSON response matching `structurepred.Response` + `gpu_seconds`

**Why plain K8s over Ray Serve / KubeRay:**
- Our demo scale (1284 pre-computed predictions + a handful of live
  dispatches) doesn't benefit from Ray's batching / scale-to-zero.
- K8s is well-documented on Nebius itself; fewer surprises.
- The FastAPI wrapper code is identical between plain-K8s and
  Ray-Serve paths — only the deployment manifest differs. Migrating
  later is hours, not days.
- One abstraction (Kubernetes) to learn, debug, and observe.

### Per-tool setup notes

**Boltz-2** (MIT/Recursion, MIT license):
- Source: github.com/jwohlwend/boltz
- Weights: HuggingFace, ~5 GB, public
- Inference: `boltz predict` CLI + Python lib
- Single A100-80GB, FP16; ~10-30s per protein-ligand pair
- No MSA dependency (uses ESM2 embeddings)

**Chai-1** (Chai Discovery, Apache 2.0):
- Source: github.com/chaidiscovery/chai-lab
- Weights: HuggingFace, public
- Inference: `chai_lab` Python lib
- Single A100-80GB; ~30-90s per pair
- No MSA dependency

**Protenix** (ByteDance, Apache 2.0):
- Source: github.com/bytedance/Protenix
- Weights: HuggingFace, public
- Inference: Python lib, similar pattern to Boltz
- Single A100-80GB; ~30-120s per pair
- No MSA dependency

All three are MSA-free, which keeps the FastAPI wrapper simple
(no need to run mmseqs2 against UniRef). They take FASTA + SMILES
in, return CIF out. The wrapper is essentially the same skeleton
for all three, with a tool-specific `predict_one(sequence, smiles)`
function.

### Kubernetes layout

```
nebius-managed-k8s (1 cluster, 3 GPU node groups)
├── namespace: orla-tools
│   ├── Deployment/Service: boltz-2     (1 pod, 1 GPU)
│   ├── Deployment/Service: chai-1      (1 pod, 1 GPU)
│   └── Deployment/Service: protenix    (1 pod, 1 GPU)
└── Ingress: orla-tools-ingress (TLS via cert-manager, public)
    ├── boltz.<domain> → boltz-2 Service
    ├── chai.<domain>  → chai-1 Service
    └── protenix.<domain> → protenix Service
```

Total YAML: 3 Deployments + 3 Services + 1 Ingress + 1 Namespace ≈
~250 lines across 4-5 files. All committed to the repo under
`demo-bio/k8s/`.

### Networking + auth

- Cluster exposes one HTTPS Ingress. TLS via cert-manager (Let's Encrypt).
- **Bearer token auth** via FastAPI middleware (5 lines per wrapper).
  Token generated once with `python3 -c 'import secrets; print(secrets.token_urlsafe(32))'`,
  stored in `.env` as `ORLA_TOOL_TOKEN`, also installed as a
  Kubernetes Secret consumed by each Deployment's env.
- Endpoint URLs: `https://boltz.<domain>/v1/tools/structure-prediction`, etc.
- orla registers each as a backend with
  `api_key_env_var: ORLA_TOOL_TOKEN`. The structurepred client reads
  the env var and sends `Authorization: Bearer <token>` — same pattern
  as the Bedrock provider.

Bearer token blocks: leaked URLs, random IP scanners, casual abuse.
Set a Nebius credit alert at $100 spend as a belt-and-suspenders
safety net.

Token rotation: rotate the K8s Secret + redeploy (2 min). orla
client picks up the new value on next dispatch.

### Cost envelope (Nebius A100-80GB ≈ $2/hr)

| Phase | Compute time | $ |
|---|---|---|
| Setup + first prediction per tool | ~3 hrs × 3 instances | $18 |
| Pre-compute all backends on 100-target subset | ~5 hrs × 3 | $30 |
| Pre-compute all backends on full 428 PoseBusters | ~12 hrs × 3 | $72 |
| Baseline + mapper runs (re-using cached predictions) | ~negligible | $0 |
| **Total** | | **~$120** |

Well under the $5K budget. The rest can fund AF3 if we add it later,
or just sit as runway.

## 4. PoseBusters integration

`demo-bio/` already has the agent / mapper / dashboard skeleton. The
structure-prediction demo lives alongside:

```
demo-bio/
  src/demo_bio/
    structure/
      __init__.py
      posebusters.py    # dataset loader + grader
      agent.py          # 3-stage structure agent
      runner.py         # batch eval
  scripts/
    setup-structure.sh  # register the 3 tool backends with orla
```

### Dataset loader

PoseBusters releases the benchmark as a `posebusters` pip package
plus a public dataset on Zenodo (the 428 protein-ligand pairs with
crystal-structure ground truth). Loader returns:

```python
@dataclass(frozen=True)
class PoseBustersTarget:
    pdb_id: str
    protein_fasta: str
    ligand_smiles: str
    reference_structure_pdb: bytes  # ground truth coords
    protein_family: str | None  # if known
```

### Grader

```python
from posebusters import PoseBusters

def grade(predicted_cif: str, reference_pdb: bytes) -> GradeResult:
    """Returns pass/fail + RMSD + per-check details."""
    pb = PoseBusters(config="dock")
    df = pb.bust(...)
    return GradeResult(passed=bool(df.iloc[0].all()), rmsd=..., details={...})
```

### Agent shape

Three stages, one of which is the tool call:

| Stage | What the LLM does | Backend kind |
|---|---|---|
| `pb-classify-target` | Read FASTA + SMILES, label target type (kinase/GPCR/...) for the mapper context | `llm` |
| `pb-predict-structure` | (no LLM) — direct tool dispatch | **`tool`** |
| `pb-grade` | Run PoseBusters checks, emit pass/fail + RMSD | (Python + LLM for diagnosis) |

The mapper routes `pb-predict-structure` between Boltz-2, Chai-1,
and Protenix. Stage context (target family from stage 1) gives the
bandit a feature to condition on; this is a small extension to the
mapper to support context features, or we can sidestep it by
creating one stage-id per target family (e.g., `pb-predict-kinase`,
`pb-predict-gpcr`, ...).

### Comparison harness

Three runs of N=428 each:

1. **Baseline 1**: always Boltz-2 (cheapest)
2. **Baseline 2**: always Chai-1 (different sweet spot)
3. **Baseline 3**: always Protenix (newest)
4. **Mapper**: bandit-routed

Per run, record: pass rate, mean RMSD, total cost (GPU-seconds × $),
mean latency. Mapper wins if pass rate matches or exceeds the best
single backend at lower total cost.

Pre-compute strategy: since all 3 × 428 = 1284 predictions are
deterministic given the same input, we run all of them once
upfront and cache results. The "baseline" and "mapper" runs then
just pick a cached prediction per target — instant, free.

This means the only real-time compute is the LLM stages around the
tool call, not the structure prediction itself. For the live demo,
optionally re-run a small sample without the cache to show the
real latency story.

## 5. Timeline + budget

### Week 1 — orla extension (Go)

| Day | Tasks | Hours |
|---|---|---|
| 1 | `Backend` interface split, `LLMProvider`/`ToolProvider` definitions, openai.go updated | 4 |
| 1 | Migration 0007 + 0008, sqlc query regeneration | 3 |
| 2 | `backends` registry updates + tests | 4 |
| 2 | Scheduler interface generalization | 3 |
| 3 | `internal/api/tool.go` handler | 4 |
| 3 | Telemetry struct + flush updates | 3 |
| 4 | `structurepred` package (Go HTTP client) | 4 |
| 4 | Wire into serve.go + integration tests | 4 |
| 5 | Dashboard adaptations + final cleanup | 4 |
| **Total** | | **~33 hours** |

### Week 2 — demo build

| Day | Tasks |
|---|---|
| 1 | Provision Nebius Managed K8s cluster with 3 GPU node groups. Set up Ingress + cert-manager. |
| 1 | Build Boltz-2 container image (Dockerfile + FastAPI wrapper). Push to a registry. Deploy. End-to-end smoke from local orla. |
| 2 | Build Chai-1 image + Deployment. End-to-end smoke. |
| 2 | Build Protenix image + Deployment. End-to-end smoke. |
| 3 | PoseBusters loader + grader integration. |
| 3-4 | Pre-compute all 3 backends × 428 targets (K8s Job or local loop). Cache. |
| 4 | Build structure agent (3 stages). Baseline runs. |
| 5 | Calibration + mapper run. Dashboard. |
| 5 | Writeup + analysis. |

### Optional Week 3

AlphaFold 3 (open-weights, with MSA pipeline). Adds the recognizable
name + an even-deeper-pocketed routing option. ~$200 in additional
compute. Skip if Boltz+Chai+Protenix tells a clean story.

## 6. Decisions needed before code lands

1. **`docs/orla-extension-plan.md` as the doc location** — already
   chosen, lives next to personas / proxy / storage / rl.
2. **AlphaFold 3 in scope?** I recommend no for v1. Yes if needed.
3. **Tool kind name**: `structure-prediction` vs `structure_prediction`
   vs `protein-structure`? My preference: `structure-prediction`
   (hyphenated, consistent with future `protein-ligand-docking`,
   `admet-prediction`, etc.).
4. **Per-target context feature for the mapper**: separate stage IDs
   per target family (simpler) vs context-aware bandit (cleaner)? I
   lean toward the simpler option for v1.
5. **Pre-compute predictions vs live run**: pre-compute lets the
   demo dashboard refresh instantly; live run is more impressive on
   stage. Both are achievable; the answer is "both — pre-compute is
   the source of truth for the result, live is a 1-target showcase
   button on the dashboard."
6. **Endpoint auth**: bearer token via FastAPI middleware (5 lines)
   + Nebius credit alert at $100. Token reused via orla's existing
   `api_key_env_var` plumbing. Confirmed.

## 7. What I'll do once you sign off

Open a draft PR with the Go-side changes only (no demo code yet).
That isolates the architectural change for clean review. Once that's
in, the demo build is purely additive — no more changes to orla
core needed.

The PR will be tagged `extension/tool-backends`. I'll keep it
single-purpose and small enough to review in one pass.
