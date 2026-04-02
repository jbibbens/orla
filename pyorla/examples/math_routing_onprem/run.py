"""MATH benchmark with chain-of-thought + Orla routing on self-hosted vLLM.

Uses the Hendrycks MATH dataset (Levels 1–5, seven subjects) to demonstrate
latency gains from accuracy-based routing on local GPUs.  Chain-of-thought
prompting produces 100–800+ output tokens per problem, making the TPS gap
between small and large models the dominant latency factor.

Two backends, each on its own GPU:

- **Cheap**: Qwen3 4B Instruct on GPU 1  (~200 tok/s output).
- **Strong**: Qwen3 32B on GPU 0  (~70 tok/s output).

Three evaluation modes:

- **baseline**: every problem goes to the strong model.
- **all-cheap**: every problem goes to the cheap model.
- **routed** (default): the dataset's built-in difficulty level (1-5) sets
  the accuracy floor, and Orla picks the cheapest qualifying backend under
  ``ACCURACY_POLICY_PREFER``.  Levels 1-3 → cheap, Levels 4-5 → strong.
  No triage LLM call needed — zero routing overhead.

Why MATH and not HotpotQA?  MATH chain-of-thought produces long outputs
(hundreds of tokens) where TPS differences between models dominate wall time.
HotpotQA answers are ~2 words, so all models finish in the same time
regardless of size.

Prerequisites:

- Two vLLM instances on separate GPUs::

    docker compose -f deploy/docker-compose.hotpotqa-vllm.yaml up -d

  Or manually::

    CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \\
        --model Qwen/Qwen3-32B --port 8000 --gpu-memory-utilization 0.90 \\
        --max-model-len 8192 --enable-auto-tool-choice --tool-call-parser hermes &
    CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server \\
        --model Qwen/Qwen3-4B-Instruct-2507 --port 8001 --gpu-memory-utilization 0.90 \\
        --max-model-len 8192 --enable-auto-tool-choice --tool-call-parser hermes &

- ``orla`` on PATH (or ``ORLA_BIN``).
- Example extras::

    uv sync --group examples

Run from the ``pyorla`` directory::

    uv run python examples/math_routing_onprem/run.py --mode baseline --limit 50
    uv run python examples/math_routing_onprem/run.py --mode all-cheap --limit 50
    uv run python examples/math_routing_onprem/run.py --mode routed --limit 50
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import time
from dataclasses import dataclass, field
from typing import Annotated, Any

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from pyorla import (
    LLMBackend,
    OrlaBinaryNotFoundError,
    OrlaClient,
    OrlaError,
    Stage,
    orla_runtime,
)
from pyorla.types import ACCURACY_POLICY_PREFER, CostModel

try:
    from datasets import load_dataset
except ImportError as exc:
    raise SystemExit(
        "This example requires Hugging Face `datasets`. Install with:\n"
        "  uv sync --group examples\n"
        "then re-run."
    ) from exc

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_CHEAP_ENDPOINT = "http://localhost:8001/v1"
DEFAULT_STRONG_ENDPOINT = "http://localhost:8000/v1"

DEFAULT_CHEAP_MODEL = "openai:Qwen/Qwen3-4B-Instruct-2507"
DEFAULT_STRONG_MODEL = "openai:Qwen/Qwen3-32B"

SOLVE_SYSTEM_PROMPT = (
    "You are an expert mathematician. Solve the following problem step by step.\n"
    "Show all your work in detail. At the very end of your response, state your\n"
    "final answer inside \\boxed{} — for example: \\boxed{42}\n"
    "Do not skip steps. Explain each transformation clearly."
)

LEVEL_TO_ACCURACY: dict[str, float] = {
    "Level 1": 0.10,
    "Level 2": 0.20,
    "Level 3": 0.30,
    "Level 4": 0.70,
    "Level 5": 0.85,
}
DEFAULT_ACCURACY = 0.85
BASELINE_ACCURACY = 0.90
HARD_LEVEL_THRESHOLD = 4


def _env(key: str, default: str) -> str:
    return os.environ.get(key, default).strip()


# ---------------------------------------------------------------------------
# MATH dataset helpers
# ---------------------------------------------------------------------------

_MATH_DATASETS = [
    ("EleutherAI/hendrycks_math", None),
    ("hendrycks/competition_math", None),
    ("DigitalLearningGmbH/MATH-lighteval", None),
]


def _load_math_dataset(split: str) -> Any:
    """Try multiple HuggingFace dataset IDs for the MATH benchmark."""
    for ds_name, config in _MATH_DATASETS:
        try:
            ds = load_dataset(ds_name, config, split=split, trust_remote_code=True)
            print(f"Loaded dataset: {ds_name} (split={split}, n={len(ds)})")
            return ds
        except Exception:  # noqa: BLE001
            continue
    raise SystemExit(
        "Could not load the MATH dataset from any known source.\n"
        "Tried: " + ", ".join(n for n, _ in _MATH_DATASETS)
    )


def _extract_gold(row: dict[str, Any]) -> str:
    """Extract the gold answer from a MATH dataset row.

    Handles both ``solution`` field (with \\boxed{}) and ``answer`` field.
    """
    if "answer" in row and row["answer"]:
        return str(row["answer"]).strip()
    solution = row.get("solution", "")
    ans = extract_boxed(solution)
    if ans is not None:
        return ans
    return solution.strip().splitlines()[-1].strip() if solution.strip() else ""


def _get_level(row: dict[str, Any]) -> str:
    """Get the MATH difficulty level from the row."""
    level = row.get("level", "")
    if isinstance(level, str):
        return level
    return str(level)


def _get_subject(row: dict[str, Any]) -> str:
    """Get the MATH subject/type from the row."""
    return str(row.get("type", row.get("subject", "")))


def load_math(
    *, split: str, start: int, limit: int
) -> list[dict[str, Any]]:
    """Load MATH items.

    Returns a list of dicts with keys ``problem``, ``gold``, ``level``,
    ``subject``, and ``solution``.
    """
    ds = _load_math_dataset(split)
    n = len(ds)
    if start < 0 or start >= n:
        raise SystemExit(
            f"--start {start} out of range for split {split!r} (len={n})"
        )
    end = n if limit == 0 else min(start + limit, n)
    items: list[dict[str, Any]] = []
    for i in range(start, end):
        row = ds[i]
        problem = row.get("problem", row.get("question", ""))
        items.append({
            "problem": str(problem),
            "gold": _extract_gold(row),
            "level": _get_level(row),
            "subject": _get_subject(row),
            "solution": str(row.get("solution", "")),
        })
    return items


# ---------------------------------------------------------------------------
# Answer extraction and comparison
# ---------------------------------------------------------------------------

def extract_boxed(text: str) -> str | None:
    r"""Extract content from the last ``\boxed{...}`` in *text*.

    Handles nested braces correctly.
    """
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return None
    start = idx + len("\\boxed{")
    depth = 1
    pos = start
    while pos < len(text) and depth > 0:
        if text[pos] == "{":
            depth += 1
        elif text[pos] == "}":
            depth -= 1
        pos += 1
    if depth != 0:
        return None
    return text[start : pos - 1].strip()


def normalize_math_answer(s: str) -> str:
    """Normalize a math answer for comparison.

    Strips whitespace, removes ``$``, ``\\text{}``, trailing periods,
    and common LaTeX formatting.
    """
    s = s.strip()
    s = s.replace("\\$", "").replace("$", "")
    s = s.replace("\\dfrac", "\\frac")
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\mathrm\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\(?:left|right)([|()[\]])", r"\1", s)
    s = s.replace("\\%", "%")
    s = s.replace("\\!", "").replace("\\ ", " ")
    s = s.replace("\\,", "")
    s = s.replace("\\cdot", "*")
    s = s.strip().rstrip(".")
    s = re.sub(r"\s+", " ", s)
    # Strip trailing percent sign so "7%" matches gold "7"
    s_no_pct = s.rstrip().rstrip("%").strip()
    try:
        val = float(s_no_pct.replace(",", ""))
        if val == int(val):
            return str(int(val))
        return f"{val:.6g}"
    except (ValueError, OverflowError):
        pass
    return s


def answers_match(predicted: str, gold: str) -> bool:
    """Check if predicted and gold answers match after normalization."""
    return normalize_math_answer(predicted) == normalize_math_answer(gold)


def extract_answer(text: str) -> str:
    r"""Extract the final answer from model output.

    Prefers ``\boxed{}``, falls back to the last non-empty line.
    """
    boxed = extract_boxed(text)
    if boxed is not None:
        return boxed
    for line in reversed(text.strip().splitlines()):
        line = line.strip()
        if line:
            return line.rstrip(".")
    return ""


# ---------------------------------------------------------------------------
# Cost tracking
# ---------------------------------------------------------------------------

def _extract_cost(msg: AIMessage) -> float:
    meta = getattr(msg, "response_metadata", None) or {}
    return float(meta.get("estimated_cost_usd", 0.0) or 0.0)


# ---------------------------------------------------------------------------
# LangGraph state
# ---------------------------------------------------------------------------

@dataclass
class MathState:
    """Messages plus accuracy floor and metadata."""

    messages: Annotated[list[AnyMessage], add_messages] = field(
        default_factory=list,
    )
    required_accuracy: float | None = None
    difficulty: str = ""


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def _user_text(state: MathState) -> str:
    for m in reversed(state.messages):
        if isinstance(m, HumanMessage) and m.content:
            return str(m.content)
    return ""


def _level_number(level: str) -> int:
    """Extract the numeric level from strings like ``Level 3``."""
    m = re.search(r"\d+", level)
    return int(m.group()) if m else 5


def _accuracy_for_item(mode: str, level: str) -> float:
    """Return the accuracy floor for a given mode and dataset level."""
    if mode == "baseline":
        return BASELINE_ACCURACY
    if mode == "all-cheap":
        return 0.0
    return LEVEL_TO_ACCURACY.get(level, DEFAULT_ACCURACY)


def build_graph(
    solve_stage: Stage,
    *,
    mode: str,
):
    """Build a LangGraph for MATH: single solve node with CoT.

    In routed mode, ``required_accuracy`` is pre-set on the state from
    the dataset's difficulty level before invoke — no triage LLM needed.
    """
    solve_llm = solve_stage.as_chat_model()

    def solve_node(state: MathState) -> dict[str, Any]:
        acc = _accuracy_for_item(mode, state.difficulty)
        solve_stage.set_accuracy(acc)
        solve_stage.set_accuracy_policy(ACCURACY_POLICY_PREFER)

        problem = _user_text(state)
        reply = solve_llm.invoke([
            SystemMessage(content=SOLVE_SYSTEM_PROMPT),
            HumanMessage(content=problem),
        ])
        return {"messages": [reply], "required_accuracy": acc}

    g = StateGraph(MathState)
    g.add_node("solve", solve_node)
    g.add_edge(START, "solve")
    g.add_edge("solve", END)
    return g.compile()


# ---------------------------------------------------------------------------
# Backend registration
# ---------------------------------------------------------------------------

def _register_backends(client: OrlaClient) -> LLMBackend:
    """Register cheap + strong on-prem backends. Returns the cheap backend."""
    api_key_env = "OPENAI_API_KEY"
    cheap = LLMBackend(
        name="math-cheap",
        endpoint=_env("MATH_CHEAP_ENDPOINT", DEFAULT_CHEAP_ENDPOINT),
        type="openai",
        model_id=_env("MATH_CHEAP_MODEL", DEFAULT_CHEAP_MODEL),
        api_key_env_var=api_key_env,
        quality=0.30,
        cost_model=CostModel(
            input_cost_per_mtoken=0.02,
            output_cost_per_mtoken=0.02,
        ),
    )
    strong = LLMBackend(
        name="math-strong",
        endpoint=_env("MATH_STRONG_ENDPOINT", DEFAULT_STRONG_ENDPOINT),
        type="openai",
        model_id=_env("MATH_STRONG_MODEL", DEFAULT_STRONG_MODEL),
        api_key_env_var=api_key_env,
        quality=0.90,
        cost_model=CostModel(
            input_cost_per_mtoken=0.05,
            output_cost_per_mtoken=0.15,
        ),
    )
    for b in (cheap, strong):
        client.register_backend(b)
    return cheap


# ---------------------------------------------------------------------------
# Cost helpers
# ---------------------------------------------------------------------------

def _triage_solve_costs(
    out: dict[str, Any], _mode: str
) -> tuple[float, float]:
    """Return ``(triage_cost, solve_cost)``.  Triage is always 0 (level-based)."""
    ai_msgs = [m for m in out["messages"] if isinstance(m, AIMessage)]
    if ai_msgs:
        return 0.0, _extract_cost(ai_msgs[-1])
    return 0.0, 0.0


# ---------------------------------------------------------------------------
# Output token estimation
# ---------------------------------------------------------------------------

def _estimate_output_tokens(msg: AIMessage) -> int:
    """Estimate output token count from response metadata or word count."""
    meta = getattr(msg, "response_metadata", None) or {}
    usage = meta.get("token_usage") or meta.get("usage") or {}
    if isinstance(usage, dict):
        out_tok = usage.get("completion_tokens") or usage.get("output_tokens")
        if out_tok is not None:
            return int(out_tok)
    text = str(msg.content)
    return int(len(text.split()) * 1.3)


# ---------------------------------------------------------------------------
# Run loop and scoring
# ---------------------------------------------------------------------------

def run_benchmark(
    client: OrlaClient,
    items: list[dict[str, Any]],
    *,
    mode: str,
    split: str,
    start: int,
    output_csv: str | None,
) -> None:
    """Run the solve pipeline on each MATH item and score."""
    cheap_be = _register_backends(client)

    solve_stage = Stage("math-solve", cheap_be)
    solve_stage.client = client
    solve_stage.set_temperature(0.0)
    solve_stage.set_max_tokens(2048)

    agent = build_graph(solve_stage, mode=mode)

    total = len(items)
    total_correct = 0
    total_cost_usd = 0.0
    total_wall_ms = 0.0
    total_output_tokens = 0
    difficulty_counts: dict[str, int] = {"easy": 0, "hard": 0}
    level_correct: dict[str, list[bool]] = {}

    csv_file = None
    csv_writer: csv.DictWriter | None = None
    if output_csv:
        csv_file = open(output_csv, "w", newline="", encoding="utf-8")  # noqa: SIM115
        fieldnames = (
            "split",
            "dataset_index",
            "run_index",
            "mode",
            "level",
            "subject",
            "gold",
            "predicted",
            "correct",
            "difficulty",
            "routing_accuracy",
            "output_tokens",
            "triage_cost_usd",
            "solve_cost_usd",
            "cost_usd",
            "wall_clock_ms",
            "problem",
        )
        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        csv_writer.writeheader()

    try:
        for idx, item in enumerate(items):
            problem = item["problem"]
            gold = item["gold"]
            level = item["level"]
            subject = item["subject"]

            print(f"\n{'=' * 60}")
            print(f"[{idx + 1}/{total}]  level={level}  subject={subject}")
            print(f"{'=' * 60}")
            print(problem[:200] + ("..." if len(problem) > 200 else ""))

            difficulty = (
                "easy"
                if _level_number(level) < HARD_LEVEL_THRESHOLD
                else "hard"
            )

            t0 = time.monotonic()
            out = agent.invoke({
                "messages": [HumanMessage(content=problem)],
                "difficulty": level,
            })
            wall_ms = (time.monotonic() - t0) * 1000.0

            last_msg = out["messages"][-1]
            reply_text = (
                str(last_msg.content)
                if isinstance(last_msg, AIMessage)
                else str(last_msg)
            )
            predicted = extract_answer(reply_text)
            correct = answers_match(predicted, gold)

            out_tokens = (
                _estimate_output_tokens(last_msg)
                if isinstance(last_msg, AIMessage)
                else int(len(reply_text.split()) * 1.3)
            )
            total_output_tokens += out_tokens

            triage_c, solve_c = _triage_solve_costs(out, mode)
            item_cost = triage_c + solve_c
            total_cost_usd += item_cost
            total_correct += int(correct)
            total_wall_ms += wall_ms

            difficulty_counts[difficulty] = (
                difficulty_counts.get(difficulty, 0) + 1
            )

            level_correct.setdefault(level, []).append(correct)

            status = "CORRECT" if correct else "WRONG"
            cost_str = f"cost=${item_cost:.6f}" if item_cost > 0 else ""
            diff_str = f"difficulty={difficulty}" if difficulty else ""
            print(f"  gold={gold!r}  predicted={predicted!r}  {status}")
            print(
                f"  {diff_str}  out_tokens~{out_tokens}  "
                f"{cost_str}  wall={wall_ms:.0f}ms"
            )

            if csv_writer is not None:
                acc = out.get("required_accuracy")
                if acc is None:
                    if mode == "baseline":
                        acc = BASELINE_ACCURACY
                    elif mode == "all-cheap":
                        acc = 0.0
                csv_writer.writerow({
                    "split": split,
                    "dataset_index": start + idx,
                    "run_index": idx,
                    "mode": mode,
                    "level": level,
                    "subject": subject,
                    "gold": gold,
                    "predicted": predicted,
                    "correct": 1 if correct else 0,
                    "difficulty": difficulty,
                    "routing_accuracy": (
                        f"{acc:.4f}" if acc is not None else ""
                    ),
                    "output_tokens": out_tokens,
                    "triage_cost_usd": f"{triage_c:.8f}",
                    "solve_cost_usd": f"{solve_c:.8f}",
                    "cost_usd": f"{item_cost:.8f}",
                    "wall_clock_ms": f"{wall_ms:.1f}",
                    "problem": problem.strip()[:500],
                })

    finally:
        if csv_file is not None:
            csv_file.close()

    acc_pct = 100.0 * total_correct / total if total else 0.0
    avg_cost = total_cost_usd / total if total else 0.0
    avg_wall = total_wall_ms / total if total else 0.0
    avg_tokens = total_output_tokens / total if total else 0.0
    print(f"\n{'=' * 60}")
    print(f"  Mode:         {mode}")
    print(f"  Items:        {total}")
    print(f"  Correct:      {total_correct}/{total}  ({acc_pct:.1f}%)")
    print(f"  Total cost:   ${total_cost_usd:.6f}")
    print(f"  Avg cost:     ${avg_cost:.6f} / item")
    print(f"  Avg latency:  {avg_wall:.0f} ms / item")
    print(f"  Avg out tok:  {avg_tokens:.0f} / item")
    print(f"  Routing:      {dict(difficulty_counts)}")
    print("  Accuracy by level:")
    for lev in sorted(level_correct.keys()):
        results = level_correct[lev]
        c = sum(results)
        n = len(results)
        print(f"    {lev}: {c}/{n} ({100.0 * c / n:.0f}%)")
    if output_csv:
        print(f"  CSV:          {output_csv}")
    print(f"{'=' * 60}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "MATH benchmark with chain-of-thought + Orla accuracy routing "
            "on self-hosted vLLM.  Produces long outputs per problem, "
            "making TPS differences between models the dominant latency factor."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("baseline", "all-cheap", "routed"),
        default="routed",
        help=(
            "baseline: always strong model. "
            "all-cheap: always cheap model. "
            "routed: LLM triage → Orla picks cheapest qualifying backend "
            "(default)."
        ),
    )
    parser.add_argument(
        "--split",
        choices=("train", "test"),
        default="test",
        help="MATH split (default: test).",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="0-based index into the split.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help=(
            "Number of consecutive examples after --start (default: 5). "
            "Use 0 for the rest of the split."
        ),
    )
    parser.add_argument(
        "--output-csv",
        default="results.csv",
        metavar="PATH",
        help=(
            "Write per-example results (UTF-8). "
            "Default: results.csv. Empty string to disable."
        ),
    )
    args = parser.parse_args()

    if args.limit < 0:
        raise SystemExit("--limit must be >= 0 (0 means through end of split)")
    if args.start < 0:
        raise SystemExit("--start must be >= 0")

    items = load_math(split=args.split, start=args.start, limit=args.limit)

    if not _env("OPENAI_API_KEY", ""):
        os.environ["OPENAI_API_KEY"] = "dummy"

    out_csv = (args.output_csv or "").strip()
    csv_path = out_csv if out_csv else None

    try:
        with orla_runtime(quiet=True) as client:
            run_benchmark(
                client,
                items,
                mode=args.mode,
                split=args.split,
                start=args.start,
                output_csv=csv_path,
            )
    except OrlaBinaryNotFoundError as exc:
        raise SystemExit(
            f"{exc}\n"
            "Install Orla, put `orla` on PATH, or set ORLA_BIN to the "
            "binary path."
        ) from exc
    except OrlaError as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
