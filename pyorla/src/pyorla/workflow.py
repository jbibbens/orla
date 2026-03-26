"""Workflow — DAG of Stages with dependency-aware scheduling."""

from __future__ import annotations

import random
import string
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from typing import Any

from pyorla.client import OrlaClient
from pyorla.stage import Stage, StageResultData
from pyorla.types import InferenceResponse, Message


DEFAULT_MAX_AGENT_LOOP_TURNS = 100


def _random_workflow_id() -> str:
    return (
        "".join(random.choices(string.ascii_lowercase, k=4))
        + "-"
        + "".join(random.choices(string.ascii_lowercase, k=4))
    )


class Workflow:
    """A DAG of Stages with dependency-aware scheduling."""

    def __init__(self, client: OrlaClient) -> None:
        self.client = client
        self._stages: dict[str, Stage] = {}
        self._dependencies: dict[str, list[str]] = {}

    def add_stage(self, stage: Stage) -> None:
        """Register a stage in the workflow DAG. Sets ``stage.client``."""
        if stage.id in self._stages:
            raise ValueError(f"stage {stage.id!r} already exists")
        stage.client = self.client
        self._stages[stage.id] = stage

    def add_dependency(self, stage_id: str, depends_on_stage_id: str) -> None:
        """Declare that *stage_id* depends on *depends_on_stage_id*."""
        if stage_id not in self._stages:
            raise ValueError(f"stage {stage_id!r} not found")
        if depends_on_stage_id not in self._stages:
            raise ValueError(f"dependency stage {depends_on_stage_id!r} not found")
        self._dependencies.setdefault(stage_id, []).append(depends_on_stage_id)

    def stages(self) -> dict[str, Stage]:
        return dict(self._stages)

    # ------------------------------------------------------------------
    # Execute the DAG
    # ------------------------------------------------------------------

    def execute(self) -> dict[str, StageResultData]:
        """Execute the workflow with dependency-aware parallelism.

        Returns results keyed by stage ID.
        """
        if not self._stages:
            return {}

        workflow_id = _random_workflow_id()
        for s in self._stages.values():
            s.set_workflow_id(workflow_id)

        remaining_deps: dict[str, int] = {}
        dependents: dict[str, list[str]] = {}
        for sid in self._stages:
            deps = self._dependencies.get(sid, [])
            remaining_deps[sid] = len(deps)
            for dep_id in deps:
                dependents.setdefault(dep_id, []).append(sid)

        results: dict[str, StageResultData] = {}

        ready = [sid for sid, count in remaining_deps.items() if count == 0]

        with ThreadPoolExecutor() as pool:
            futures: dict[Future[tuple[str, StageResultData]], str] = {}

            def _submit(stage_id: str) -> None:
                snapshot = dict(results)
                fut: Future[tuple[str, StageResultData]] = pool.submit(
                    self._execute_stage_in_dag, stage_id, snapshot
                )
                futures[fut] = stage_id

            for sid in ready:
                _submit(sid)

            while futures:
                for fut in as_completed(futures):
                    stage_id = futures.pop(fut)
                    sid, result = fut.result()
                    results[sid] = result

                    for dep in dependents.get(sid, []):
                        remaining_deps[dep] -= 1
                        if remaining_deps[dep] == 0:
                            _submit(dep)
                    break  # restart as_completed after submitting new work

        self._notify_workflow_complete(workflow_id)
        return results

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _execute_stage_in_dag(
        self,
        stage_id: str,
        dep_results: dict[str, StageResultData],
    ) -> tuple[str, StageResultData]:
        stage = self._stages[stage_id]
        if stage.execution_mode == "agent_loop":
            return stage_id, self._execute_agent_loop(stage, dep_results)
        return stage_id, self._execute_single_shot(stage, dep_results)

    def _execute_single_shot(
        self, stage: Stage, dep_results: dict[str, StageResultData]
    ) -> StageResultData:
        if stage.messages_builder is not None:
            msgs = stage.messages_builder(dep_results)
            if stage.stream:
                stream = stage.execute_stream_with_messages(msgs)
                resp = stage.consume_stream(stream)
                return StageResultData(response=resp, messages=msgs)
            resp = stage.execute_with_messages(msgs)
            return StageResultData(response=resp, messages=msgs)

        prompt = stage.prompt
        if stage.prompt_builder is not None:
            prompt = stage.prompt_builder(dep_results)
        if not prompt:
            raise ValueError(f"stage {stage.name!r}: prompt is empty")

        if stage.stream:
            stream = stage.execute_stream(prompt)
            resp = stage.consume_stream(stream)
            return StageResultData(response=resp)
        resp = stage.execute(prompt)
        return StageResultData(response=resp)

    def _execute_agent_loop(
        self, stage: Stage, dep_results: dict[str, StageResultData]
    ) -> StageResultData:
        messages: list[Message]

        if stage.messages_builder is not None:
            messages = stage.messages_builder(dep_results)
        else:
            prompt = stage.prompt
            if stage.prompt_builder is not None:
                prompt = stage.prompt_builder(dep_results)
            if not prompt:
                raise ValueError(f"stage {stage.name!r}: prompt is empty")
            messages = [Message(role="user", content=prompt)]

        max_turns = stage.max_turns if stage.max_turns > 0 else DEFAULT_MAX_AGENT_LOOP_TURNS
        last_resp: InferenceResponse | None = None

        for _ in range(max_turns):
            resp = stage.execute_with_messages(messages)
            last_resp = resp

            messages.append(
                Message(role="assistant", content=resp.content, tool_calls=resp.tool_calls)
            )

            if not resp.tool_calls:
                break

            tool_results = stage.run_tool_calls_in_response(resp)
            for tr in tool_results:
                msg_dict = tr.to_message_dict()
                messages.append(
                    Message(
                        role="tool",
                        content=msg_dict["content"],
                        tool_call_id=msg_dict.get("tool_call_id", ""),
                        tool_name=msg_dict.get("tool_name", ""),
                    )
                )

        return StageResultData(response=last_resp, messages=messages)

    def _notify_workflow_complete(self, workflow_id: str) -> None:
        backends: set[str] = set()
        for stage in self._stages.values():
            if stage.backend is not None and stage.backend.name:
                backends.add(stage.backend.name)
        if not backends:
            return
        try:
            self.client.workflow_complete(workflow_id, list(backends))
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> Workflow:
        return self

    def __exit__(self, *_: Any) -> None:
        pass
