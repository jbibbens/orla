"""Stage mapping — pre-execution planning and backend assignment."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pyorla.types import LLMBackend, StructuredOutputRequest

if TYPE_CHECKING:
    from pyorla.stage import Stage


@dataclass
class StageAssignment:
    """Backend and inference parameters assigned to a stage."""

    backend: LLMBackend | None = None
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    response_format: StructuredOutputRequest | None = None


@dataclass
class StageMappingInput:
    """All stages that need assignment plus available backends."""

    stages: list[Stage] = field(default_factory=list)
    backends: list[LLMBackend] = field(default_factory=list)


@dataclass
class StageMappingOutput:
    """Per-stage assignments keyed by stage ID."""

    assignments: dict[str, StageAssignment] = field(default_factory=dict)


class StageMapping(ABC):
    """Base class for stage mapping strategies."""

    @abstractmethod
    def map(self, input: StageMappingInput) -> StageMappingOutput: ...


class ExplicitStageMapping(StageMapping):
    """Validates every stage already has a backend assigned."""

    def map(self, input: StageMappingInput) -> StageMappingOutput:
        output = StageMappingOutput()
        for stage in input.stages:
            if stage.backend is None:
                raise ValueError(
                    f"stage {stage.name!r} ({stage.id}) has no backend assigned"
                )
            output.assignments[stage.id] = StageAssignment(
                backend=stage.backend,
                max_tokens=stage.max_tokens,
                temperature=stage.temperature,
                top_p=stage.top_p,
                response_format=stage.response_format,
            )
        return output


def apply_stage_mapping_output(
    stages: list[Stage], output: StageMappingOutput
) -> None:
    """Apply mapping output to stages."""
    for stage in stages:
        assignment = output.assignments.get(stage.id)
        if assignment is None:
            continue
        if assignment.backend is not None:
            stage.backend = assignment.backend
        if assignment.max_tokens is not None:
            stage.max_tokens = assignment.max_tokens
        if assignment.temperature is not None:
            stage.temperature = assignment.temperature
        if assignment.top_p is not None:
            stage.top_p = assignment.top_p
        if assignment.response_format is not None:
            stage.response_format = assignment.response_format
