"""Memory and KV-cache policy types for multi-stage workflows."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from pyorla.types import CACHE_POLICY_AUTO, CACHE_POLICY_FLUSH, CACHE_POLICY_PRESERVE


@dataclass
class CacheEvent:
    """Describes a stage transition for the MemoryPolicy to evaluate."""

    prev_stage_backend: str = ""
    prev_stage_model: str = ""
    next_stage_backend: str = ""
    next_stage_model: str = ""
    delta_tokens: int = 0
    total_tokens: int = 0
    transition_type: str = ""  # "stage" | "workflow_complete"


class MemoryPolicy(ABC):
    """Determines cache actions at workflow level."""

    @abstractmethod
    def decide(self, event: CacheEvent) -> str:
        """Return one of CACHE_POLICY_PRESERVE, CACHE_POLICY_FLUSH, CACHE_POLICY_AUTO."""
        ...


class PreserveOnSmallIncrementPolicy(MemoryPolicy):
    """Preserve KV cache when context delta is small and backend hasn't changed."""

    def __init__(self, threshold_tokens: int = 256) -> None:
        self.threshold_tokens = max(threshold_tokens, 1)

    def decide(self, event: CacheEvent) -> str:
        if (
            event.prev_stage_backend != event.next_stage_backend
            or event.prev_stage_model != event.next_stage_model
        ):
            return CACHE_POLICY_AUTO
        if not event.prev_stage_backend:
            return CACHE_POLICY_AUTO
        if event.delta_tokens <= self.threshold_tokens:
            return CACHE_POLICY_PRESERVE
        return CACHE_POLICY_AUTO


class FlushAtBoundaryPolicy(MemoryPolicy):
    """Flush cache at workflow boundaries and backend switches."""

    def decide(self, event: CacheEvent) -> str:
        if event.transition_type == "workflow_complete":
            return CACHE_POLICY_FLUSH
        if event.prev_stage_backend and (
            event.prev_stage_backend != event.next_stage_backend
            or event.prev_stage_model != event.next_stage_model
        ):
            return CACHE_POLICY_FLUSH
        return CACHE_POLICY_AUTO


class DefaultMemoryPolicy(MemoryPolicy):
    """Composed policy: preserve-on-small-increment then flush-at-boundary."""

    def __init__(self, preserve_threshold: int = 256) -> None:
        self._policies: list[MemoryPolicy] = [
            PreserveOnSmallIncrementPolicy(preserve_threshold),
            FlushAtBoundaryPolicy(),
        ]

    def decide(self, event: CacheEvent) -> str:
        for policy in self._policies:
            result = policy.decide(event)
            if result != CACHE_POLICY_AUTO:
                return result
        return CACHE_POLICY_AUTO
