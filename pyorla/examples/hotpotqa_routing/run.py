"""HotpotQA evaluation with LangGraph + Orla accuracy routing.

Uses a multi-stage decomposition pipeline to answer multi-hop questions from
the HotpotQA distractor benchmark.  Each question passes through up to four
LLM stages (triage, decompose, answer sub-questions, synthesize), producing
5-7 inference calls whose per-call latency differences compound across the
pipeline.  Evaluation follows the official HotpotQA protocol (answer EM and
token-level F1 after normalization).

Three modes compare cost, accuracy, and latency:

- **baseline**: all stages route to the strong model (high accuracy, high cost).
- **all-cheap**: all stages route to the cheap model (low cost, lower accuracy).
- **routed** (default): a triage node classifies difficulty, then sets
  ``accuracy`` so Orla picks the cheapest qualifying backend under
  ``ACCURACY_POLICY_PREFER`` for the remaining stages.

Model tiers (all open-weight, Bedrock Mantle defaults, overridable via env vars):

- **Cheap**: Ministral 3B  (486 t/s on Bedrock, very low cost).
- **Mid**: Qwen3 32B dense (257 t/s, strong reasoning at moderate cost).
- **Strong**: Qwen3 235B A22B (84 t/s, frontier-class MoE reasoning).

Override ``HOTPOTQA_CHEAP_MODEL``, ``HOTPOTQA_MID_MODEL``,
``HOTPOTQA_STRONG_MODEL``, and ``OPENAI_BASE_URL`` to point at a local
vLLM/Ollama endpoint.

Prerequisites:

- ``orla`` on PATH (or ``ORLA_BIN``); this script starts a local daemon via
  ``orla_runtime()``.
- ``OPENAI_API_KEY`` for the Bedrock Mantle OpenAI-compatible API (or for
  your local endpoint).
- Example extras (Hugging Face ``datasets``)::

    uv sync --group examples

Run from the ``pyorla`` directory::

    uv run python examples/hotpotqa_routing/run.py --mode routed --limit 20
    uv run python examples/hotpotqa_routing/run.py --mode baseline --limit 20
    uv run python examples/hotpotqa_routing/run.py --mode all-cheap --limit 20

Use ``--limit 0`` to evaluate the entire split from ``--start`` (full
validation set: ``--split validation --start 0 --limit 0``).

Per-example metrics are written to ``results.csv`` by default (override with
``--output-csv``; empty value disables).  Columns include ``cost_usd``,
per-stage costs, ``answer_em``, ``answer_f1``, ``wall_clock_ms``, and
``difficulty`` for plotting.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import string
import time
from collections import Counter
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

MANTLE_BASE_URL = "https://bedrock-mantle.us-east-2.api.aws/v1"
DEFAULT_CHEAP_MODEL = "openai:mistral.ministral-3-3b-instruct"
DEFAULT_MID_MODEL = "openai:qwen.qwen3-32b-v1:0"
DEFAULT_STRONG_MODEL = "openai:qwen.qwen3-235b-a22b-2507-v1:0"

TRIAGE_SYSTEM_PROMPT = (
    "You classify multi-hop questions by difficulty.\n"
    "Output ONLY one word: easy, medium, or hard.\n"
    "- easy: answer is directly stated or requires trivial lookup\n"
    "- medium: requires connecting facts from two paragraphs\n"
    "- hard: requires complex multi-step reasoning, comparison, or inference"
)

DECOMPOSE_SYSTEM_PROMPT = (
    "Break the following question into simpler sub-questions that can each "
    "be answered from a single paragraph.  Output one sub-question per line, "
    'prefixed with "Q1:", "Q2:", etc.  Use 2-3 sub-questions.'
)

SUB_ANSWER_SYSTEM_PROMPT = (
    "Using the paragraphs below, answer the following sub-question in one "
    "or two sentences.  If the answer is not in the paragraphs, say "
    '"unknown".'
)

SYNTHESIZE_SYSTEM_PROMPT = (
    "Given the original question and the answers to its sub-questions, "
    "provide the final answer.  Think step by step, then end your response "
    'with "Answer: <answer>" where <answer> is a short, specific answer '
    "(a few words at most)."
)

DIFFICULTY_TO_ACCURACY: dict[str, float] = {
    "easy": 0.30,
    "medium": 0.60,
    "hard": 0.85,
}
DEFAULT_ACCURACY = 0.85
BASELINE_ACCURACY = 0.90

MAX_SUB_QUESTIONS = 3


def _env(key: str, default: str) -> str:
    return os.environ.get(key, default).strip()


# ---------------------------------------------------------------------------
# HotpotQA dataset helpers
# ---------------------------------------------------------------------------

def format_context(context: dict[str, Any]) -> str:
    """Format the 10 distractor paragraphs into readable text."""
    titles = context["title"]
    sentences_list = context["sentences"]
    parts: list[str] = []
    for title, sentences in zip(titles, sentences_list):
        body = " ".join(s.strip() for s in sentences if s.strip())
        parts.append(f"[{title}]\n{body}")
    return "\n\n".join(parts)


def load_hotpotqa(
    *, split: str, start: int, limit: int
) -> list[dict[str, Any]]:
    """Load HotpotQA distractor items.

    Returns a list of dicts with keys ``id``, ``question``, ``answer``,
    ``type``, ``level``, and ``context_str`` (pre-formatted paragraphs).
    """
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split=split)
    n = len(ds)
    if start < 0 or start >= n:
        raise SystemExit(
            f"--start {start} out of range for split {split!r} (len={n})"
        )
    end = n if limit == 0 else min(start + limit, n)
    items: list[dict[str, Any]] = []
    for i in range(start, end):
        row = ds[i]
        items.append({
            "id": row["id"],
            "question": row["question"],
            "answer": row["answer"],
            "type": row["type"],
            "level": row["level"],
            "context_str": format_context(row["context"]),
        })
    return items


# ---------------------------------------------------------------------------
# Official HotpotQA evaluation (ported from hotpot_evaluate_v1.py)
# ---------------------------------------------------------------------------

def normalize_answer(s: str) -> str:
    """Lowercase, strip articles / punctuation, collapse whitespace."""
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in string.punctuation)
    return " ".join(s.split())


def exact_match_score(prediction: str, ground_truth: str) -> bool:
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def f1_score(prediction: str, ground_truth: str) -> float:
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    if (
        normalized_prediction in ("yes", "no", "noanswer")
        and normalized_prediction != normalized_ground_truth
    ):
        return 0.0
    if (
        normalized_ground_truth in ("yes", "no", "noanswer")
        and normalized_prediction != normalized_ground_truth
    ):
        return 0.0

    pred_tokens = normalized_prediction.split()
    gold_tokens = normalized_ground_truth.split()
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return (2 * precision * recall) / (precision + recall)


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

_ANSWER_RE = re.compile(r"[Aa]nswer\s*:\s*(.+)")


def extract_answer(text: str) -> str:
    """Extract the final answer from model output.

    Looks for ``Answer: <text>`` first, falls back to the last non-empty
    line.
    """
    for line in reversed(text.strip().splitlines()):
        m = _ANSWER_RE.search(line)
        if m:
            return m.group(1).strip().rstrip(".")
    for line in reversed(text.strip().splitlines()):
        line = line.strip()
        if line:
            return line.rstrip(".")
    return ""


# ---------------------------------------------------------------------------
# Sub-question parsing
# ---------------------------------------------------------------------------

_SUBQ_RE = re.compile(r"Q\d+\s*:\s*(.+)")


def parse_sub_questions(text: str) -> list[str]:
    """Parse ``Q1: ...`` lines from the decompose output."""
    questions: list[str] = []
    for line in text.strip().splitlines():
        m = _SUBQ_RE.match(line.strip())
        if m:
            questions.append(m.group(1).strip())
    if not questions:
        questions = [
            line.strip()
            for line in text.strip().splitlines()
            if line.strip()
        ]
    return questions[:MAX_SUB_QUESTIONS]


# ---------------------------------------------------------------------------
# Cost tracking
# ---------------------------------------------------------------------------

def _extract_cost(msg: AIMessage) -> float:
    """Read Orla's ``estimated_cost_usd`` from LangChain response_metadata."""
    meta = getattr(msg, "response_metadata", None) or {}
    return float(meta.get("estimated_cost_usd", 0.0) or 0.0)


# ---------------------------------------------------------------------------
# LangGraph state
# ---------------------------------------------------------------------------

@dataclass
class HotpotState:
    """Carries messages, accuracy floor, decomposition artifacts."""

    messages: Annotated[list[AnyMessage], add_messages] = field(
        default_factory=list,
    )
    required_accuracy: float | None = None
    difficulty: str = ""
    context_str: str = ""
    question: str = ""
    sub_questions: list[str] = field(default_factory=list)
    sub_answers: list[str] = field(default_factory=list)
    num_subquestions: int = 0


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def _accuracy_for_mode(mode: str, state: HotpotState) -> float:
    if mode == "baseline":
        return BASELINE_ACCURACY
    if mode == "all-cheap":
        return 0.0
    return (
        state.required_accuracy
        if state.required_accuracy is not None
        else DEFAULT_ACCURACY
    )


def build_graph(
    *,
    mode: str,
    triage_stage: Stage | None = None,
    decompose_stage: Stage,
    sub_answer_stage: Stage,
    synthesize_stage: Stage,
):
    """Build a LangGraph for HotpotQA multi-stage decomposition.

    *mode* is one of ``routed``, ``baseline``, ``all-cheap``.
    """

    # -- Node 1: Triage (routed mode only) --------------------------------

    def triage_node(state: HotpotState) -> dict[str, Any]:
        assert triage_stage is not None
        triage_llm = triage_stage.as_chat_model()
        reply = triage_llm.invoke([
            SystemMessage(content=TRIAGE_SYSTEM_PROMPT),
            HumanMessage(content=state.question),
        ])
        label = str(reply.content).strip().lower().rstrip(".")
        acc = DIFFICULTY_TO_ACCURACY.get(label, DEFAULT_ACCURACY)
        return {
            "messages": [reply],
            "required_accuracy": acc,
            "difficulty": label,
        }

    # -- Node 2: Decompose ------------------------------------------------

    def decompose_node(state: HotpotState) -> dict[str, Any]:
        acc = _accuracy_for_mode(mode, state)
        decompose_stage.set_accuracy(acc)
        decompose_stage.set_accuracy_policy(ACCURACY_POLICY_PREFER)

        llm = decompose_stage.as_chat_model()
        reply = llm.invoke([
            SystemMessage(content=DECOMPOSE_SYSTEM_PROMPT),
            HumanMessage(content=state.question),
        ])
        text = str(reply.content)
        subs = parse_sub_questions(text)
        if not subs:
            subs = [state.question]
        return {
            "messages": [reply],
            "sub_questions": subs,
            "num_subquestions": len(subs),
        }

    # -- Node 3: Answer sub-questions -------------------------------------

    def answer_subqs_node(state: HotpotState) -> dict[str, Any]:
        acc = _accuracy_for_mode(mode, state)
        sub_answer_stage.set_accuracy(acc)
        sub_answer_stage.set_accuracy_policy(ACCURACY_POLICY_PREFER)

        llm = sub_answer_stage.as_chat_model()
        answers: list[str] = []
        reply_msgs: list[AIMessage] = []
        for sq in state.sub_questions:
            prompt = (
                f"{state.context_str}\n\nSub-question: {sq}"
            )
            reply = llm.invoke([
                SystemMessage(content=SUB_ANSWER_SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ])
            answers.append(str(reply.content).strip())
            reply_msgs.append(reply)
        return {
            "messages": reply_msgs,
            "sub_answers": answers,
        }

    # -- Node 4: Synthesize -----------------------------------------------

    def synthesize_node(state: HotpotState) -> dict[str, Any]:
        acc = _accuracy_for_mode(mode, state)
        synthesize_stage.set_accuracy(acc)
        synthesize_stage.set_accuracy_policy(ACCURACY_POLICY_PREFER)

        llm = synthesize_stage.as_chat_model()
        qa_lines: list[str] = []
        for i, (sq, sa) in enumerate(
            zip(state.sub_questions, state.sub_answers), 1
        ):
            qa_lines.append(f"Q{i}: {sq} -> {sa}")
        body = "\n".join(qa_lines)
        prompt = (
            f"Question: {state.question}\n\n"
            f"Sub-question answers:\n{body}"
        )
        reply = llm.invoke([
            SystemMessage(content=SYNTHESIZE_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ])
        return {"messages": [reply]}

    # -- Assemble graph ---------------------------------------------------

    g = StateGraph(HotpotState)
    g.add_node("decompose", decompose_node)
    g.add_node("answer_subqs", answer_subqs_node)
    g.add_node("synthesize", synthesize_node)

    if mode == "routed":
        g.add_node("triage", triage_node)
        g.add_edge(START, "triage")
        g.add_edge("triage", "decompose")
    else:
        g.add_edge(START, "decompose")

    g.add_edge("decompose", "answer_subqs")
    g.add_edge("answer_subqs", "synthesize")
    g.add_edge("synthesize", END)
    return g.compile()


# ---------------------------------------------------------------------------
# Backend registration
# ---------------------------------------------------------------------------

def _register_backends(client: OrlaClient) -> LLMBackend:
    """Register cheap / mid / strong backends.

    Returns the cheap backend, used as the nominal backend for all routed
    stages (Orla rewrites the selection via accuracy routing).
    """
    api_key_env = "OPENAI_API_KEY"
    cheap = LLMBackend(
        name="hotpot-cheap",
        endpoint=_env("OPENAI_BASE_URL", MANTLE_BASE_URL),
        type="openai",
        model_id=_env("HOTPOTQA_CHEAP_MODEL", DEFAULT_CHEAP_MODEL),
        api_key_env_var=api_key_env,
        quality=0.30,
        cost_model=CostModel(
            input_cost_per_mtoken=0.10,
            output_cost_per_mtoken=0.10,
        ),
    )
    mid = LLMBackend(
        name="hotpot-mid",
        endpoint=_env("OPENAI_BASE_URL", MANTLE_BASE_URL),
        type="openai",
        model_id=_env("HOTPOTQA_MID_MODEL", DEFAULT_MID_MODEL),
        api_key_env_var=api_key_env,
        quality=0.60,
        cost_model=CostModel(
            input_cost_per_mtoken=0.15,
            output_cost_per_mtoken=0.60,
        ),
    )
    strong = LLMBackend(
        name="hotpot-strong",
        endpoint=_env("OPENAI_BASE_URL", MANTLE_BASE_URL),
        type="openai",
        model_id=_env("HOTPOTQA_STRONG_MODEL", DEFAULT_STRONG_MODEL),
        api_key_env_var=api_key_env,
        quality=0.90,
        cost_model=CostModel(
            input_cost_per_mtoken=0.53,
            output_cost_per_mtoken=2.66,
        ),
    )
    for b in (cheap, mid, strong):
        client.register_backend(b)
    return cheap


# ---------------------------------------------------------------------------
# Cost helpers for multi-stage pipeline
# ---------------------------------------------------------------------------

def _stage_costs(
    out: dict[str, Any], mode: str
) -> tuple[float, float, float, float]:
    """Split costs across the four pipeline stages.

    Returns ``(triage, decompose, subq_total, synthesize)`` costs.
    """
    ai_msgs = [m for m in out["messages"] if isinstance(m, AIMessage)]
    if not ai_msgs:
        return 0.0, 0.0, 0.0, 0.0

    if mode == "routed":
        # triage, decompose, N sub-q answers, synthesize
        triage_cost = _extract_cost(ai_msgs[0]) if len(ai_msgs) >= 1 else 0.0
        decompose_cost = (
            _extract_cost(ai_msgs[1]) if len(ai_msgs) >= 2 else 0.0
        )
        synthesize_cost = (
            _extract_cost(ai_msgs[-1]) if len(ai_msgs) >= 3 else 0.0
        )
        subq_cost = sum(
            _extract_cost(m) for m in ai_msgs[2:-1]
        ) if len(ai_msgs) >= 4 else 0.0
    else:
        # decompose, N sub-q answers, synthesize  (no triage)
        triage_cost = 0.0
        decompose_cost = (
            _extract_cost(ai_msgs[0]) if len(ai_msgs) >= 1 else 0.0
        )
        synthesize_cost = (
            _extract_cost(ai_msgs[-1]) if len(ai_msgs) >= 2 else 0.0
        )
        subq_cost = sum(
            _extract_cost(m) for m in ai_msgs[1:-1]
        ) if len(ai_msgs) >= 3 else 0.0

    return triage_cost, decompose_cost, subq_cost, synthesize_cost


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
    """Run the multi-stage pipeline on each HotpotQA item and score."""
    cheap_be = _register_backends(client)

    def _make_stage(name: str, max_tokens: int) -> Stage:
        s = Stage(name, cheap_be)
        s.client = client
        s.set_temperature(0.0)
        s.set_max_tokens(max_tokens)
        return s

    triage_stage: Stage | None = None
    if mode == "routed":
        triage_stage = _make_stage("hotpot-triage", 8)

    decompose_stage = _make_stage("hotpot-decompose", 256)
    sub_answer_stage = _make_stage("hotpot-sub-answer", 256)
    synthesize_stage = _make_stage("hotpot-synthesize", 256)

    agent = build_graph(
        mode=mode,
        triage_stage=triage_stage,
        decompose_stage=decompose_stage,
        sub_answer_stage=sub_answer_stage,
        synthesize_stage=synthesize_stage,
    )

    total = len(items)
    total_em = 0
    total_f1 = 0.0
    total_cost_usd = 0.0
    total_wall_ms = 0.0
    difficulty_counts: dict[str, int] = {
        "easy": 0, "medium": 0, "hard": 0, "unknown": 0,
    }
    type_counts: dict[str, int] = {"bridge": 0, "comparison": 0}

    csv_file = None
    csv_writer: csv.DictWriter | None = None
    if output_csv:
        csv_file = open(output_csv, "w", newline="", encoding="utf-8")  # noqa: SIM115
        fieldnames = (
            "split",
            "dataset_index",
            "run_index",
            "mode",
            "id",
            "question_type",
            "level",
            "gold",
            "predicted",
            "answer_em",
            "answer_f1",
            "difficulty",
            "routing_accuracy",
            "num_subquestions",
            "triage_cost_usd",
            "decompose_cost_usd",
            "subq_cost_usd",
            "synthesize_cost_usd",
            "cost_usd",
            "wall_clock_ms",
            "question",
        )
        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        csv_writer.writeheader()

    try:
        for idx, item in enumerate(items):
            question = item["question"]
            gold = item["answer"]
            q_type = item["type"]
            level = item["level"]
            context_str = item["context_str"]

            print(f"\n{'=' * 60}")
            print(f"[{idx + 1}/{total}]  gold={gold!r}  type={q_type}  level={level}")
            print(f"{'=' * 60}")
            print(question.strip())

            t0 = time.monotonic()
            out = agent.invoke({
                "messages": [HumanMessage(content=question)],
                "question": question,
                "context_str": context_str,
            })
            wall_ms = (time.monotonic() - t0) * 1000.0

            last_msg = out["messages"][-1]
            reply_text = (
                str(last_msg.content)
                if isinstance(last_msg, AIMessage)
                else str(last_msg)
            )
            predicted = extract_answer(reply_text)
            em = exact_match_score(predicted, gold)
            f1 = f1_score(predicted, gold)

            triage_c, decompose_c, subq_c, synth_c = _stage_costs(out, mode)
            item_cost = triage_c + decompose_c + subq_c + synth_c
            total_cost_usd += item_cost
            total_em += int(em)
            total_f1 += f1
            total_wall_ms += wall_ms

            difficulty = out.get("difficulty", "")
            if difficulty:
                difficulty_counts[difficulty] = (
                    difficulty_counts.get(difficulty, 0) + 1
                )
            elif mode == "routed":
                difficulty_counts["unknown"] += 1
            type_counts[q_type] = type_counts.get(q_type, 0) + 1

            n_subq = out.get("num_subquestions", 0)
            status = "EM" if em else f"F1={f1:.2f}"
            cost_str = f"cost=${item_cost:.6f}" if item_cost > 0 else ""
            diff_str = f"difficulty={difficulty}" if difficulty else ""
            print(
                f"  predicted={predicted!r}  {status}  "
                f"{diff_str}  {cost_str}  wall={wall_ms:.0f}ms"
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
                    "id": item["id"],
                    "question_type": q_type,
                    "level": level,
                    "gold": gold,
                    "predicted": predicted,
                    "answer_em": 1 if em else 0,
                    "answer_f1": f"{f1:.4f}",
                    "difficulty": difficulty,
                    "routing_accuracy": f"{acc:.4f}" if acc is not None else "",
                    "num_subquestions": n_subq,
                    "triage_cost_usd": f"{triage_c:.8f}",
                    "decompose_cost_usd": f"{decompose_c:.8f}",
                    "subq_cost_usd": f"{subq_c:.8f}",
                    "synthesize_cost_usd": f"{synth_c:.8f}",
                    "cost_usd": f"{item_cost:.8f}",
                    "wall_clock_ms": f"{wall_ms:.1f}",
                    "question": question.strip(),
                })

    finally:
        if csv_file is not None:
            csv_file.close()

    em_pct = 100.0 * total_em / total if total else 0.0
    avg_f1 = total_f1 / total if total else 0.0
    avg_cost = total_cost_usd / total if total else 0.0
    avg_wall = total_wall_ms / total if total else 0.0
    print(f"\n{'=' * 60}")
    print(f"  Mode:        {mode}")
    print(f"  Items:       {total}")
    print(f"  Answer EM:   {em_pct:.1f}%  ({total_em}/{total})")
    print(f"  Answer F1:   {avg_f1:.3f}")
    print(f"  Total cost:  ${total_cost_usd:.6f}")
    print(f"  Avg cost:    ${avg_cost:.6f} / item")
    print(f"  Avg latency: {avg_wall:.0f} ms / item")
    if mode == "routed":
        print(f"  Difficulty:  {dict(difficulty_counts)}")
    print(f"  Types:       {dict(type_counts)}")
    if output_csv:
        print(f"  CSV:         {output_csv}")
    print(f"{'=' * 60}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "HotpotQA benchmark with LangGraph + Orla accuracy routing "
            "(multi-stage decomposition pipeline)."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("baseline", "all-cheap", "routed"),
        default="routed",
        help=(
            "baseline: always strong model. "
            "all-cheap: always cheap model. "
            "routed: LLM triage -> Orla picks cheapest qualifying backend "
            "(default)."
        ),
    )
    parser.add_argument(
        "--split",
        choices=("train", "validation"),
        default="validation",
        help="HotpotQA distractor split (default: validation).",
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
            "Write per-example results for plotting (UTF-8). "
            "Default: results.csv. Use an empty string to disable."
        ),
    )
    args = parser.parse_args()

    if args.limit < 0:
        raise SystemExit("--limit must be >= 0 (0 means through end of split)")
    if args.start < 0:
        raise SystemExit("--start must be >= 0")

    items = load_hotpotqa(
        split=args.split, start=args.start, limit=args.limit,
    )

    if not _env("OPENAI_API_KEY", ""):
        raise SystemExit(
            "OPENAI_API_KEY must be set so `orla serve` can call the "
            "Mantle endpoint."
        )

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
