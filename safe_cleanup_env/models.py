from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class FileRecord:
    file_id: str
    name: str
    current_path: str
    extension: str
    size_mb: float
    content_hint: str
    risk_flags: list[str]
    candidate: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExpectedDisposition:
    decision: str
    destination: str | None = None
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CleanupTask:
    task_id: str
    title: str
    difficulty: str
    instruction: str
    allowed_destinations: list[str]
    protected_roots: list[str]
    candidate_files: list[FileRecord]
    protected_inventory: list[FileRecord]
    expectations: dict[str, ExpectedDisposition]

    def to_observation(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "title": self.title,
            "difficulty": self.difficulty,
            "instruction": self.instruction,
            "candidate_files": [file.to_dict() for file in self.candidate_files],
            "protected_roots": list(self.protected_roots),
            "allowed_destinations": list(self.allowed_destinations),
            "protected_inventory": [file.to_dict() for file in self.protected_inventory],
        }


@dataclass
class CleanupStateSnapshot:
    proposed_destinations: dict[str, str] = field(default_factory=dict)
    moved_files: dict[str, str] = field(default_factory=dict)
    skipped_files: dict[str, str] = field(default_factory=dict)
    inspected_files: list[str] = field(default_factory=list)
    safety_violations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ScoreBreakdown:
    decision_accuracy: float = 0.0
    safety_score: float = 0.0
    completion: float = 0.0
    total: float = 0.0
    per_file: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SafeCleanupAction:
    action_type: str
    file_id: str | None = None
    destination: str | None = None
    reason: str | None = None

    def to_payload(self) -> dict[str, Any]:
        payload = {"action_type": self.action_type}
        if self.file_id is not None:
            payload["file_id"] = self.file_id
        if self.destination is not None:
            payload["destination"] = self.destination
        if self.reason is not None:
            payload["reason"] = self.reason
        return payload


@dataclass
class SafeCleanupObservation:
    task_id: str
    title: str
    difficulty: str
    instruction: str
    candidate_files: list[dict[str, Any]]
    protected_roots: list[str]
    allowed_destinations: list[str]
    protected_inventory: list[dict[str, Any]]
    remaining_file_ids: list[str]
    feedback: str
    current_state: dict[str, Any]
    step_count: int
    max_steps: int
    done: bool
    score_so_far: float
    action_log: list[dict[str, Any]]

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "SafeCleanupObservation":
        return cls(**payload)


@dataclass
class SafeCleanupState:
    task_id: str
    step_count: int
    max_steps: int
    done: bool
    total_reward: float
    state_snapshot: dict[str, Any]
    action_log: list[dict[str, Any]]

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "SafeCleanupState":
        return cls(**payload)


@dataclass
class SafeCleanupStepResult:
    observation: SafeCleanupObservation
    reward: float
    done: bool
    info: dict[str, Any]

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "SafeCleanupStepResult":
        return cls(
            observation=SafeCleanupObservation.from_payload(payload["observation"]),
            reward=payload["reward"],
            done=payload["done"],
            info=payload["info"],
        )
