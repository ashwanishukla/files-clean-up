from __future__ import annotations

from typing import Any

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class FilesCleanUpAction(Action):
    action_type: str = Field(
        ..., description="One of inspect_file, propose_destination, move_file, skip_file, submit"
    )
    file_id: str | None = Field(default=None, description="Candidate file id for file actions")
    destination: str | None = Field(default=None, description="Allowed destination folder")
    reason: str | None = Field(default=None, description="Reason when skipping a file")


class FilesCleanUpObservation(Observation):
    task_id: str = Field(default="", description="Task id")
    title: str = Field(default="", description="Task title")
    difficulty: str = Field(default="", description="Task difficulty")
    instruction: str = Field(default="", description="Agent instruction")
    candidate_files: list[dict[str, Any]] = Field(default_factory=list)
    protected_roots: list[str] = Field(default_factory=list)
    allowed_destinations: list[str] = Field(default_factory=list)
    protected_inventory: list[dict[str, Any]] = Field(default_factory=list)
    remaining_file_ids: list[str] = Field(default_factory=list)
    feedback: str = Field(default="")
    current_state: dict[str, Any] = Field(default_factory=dict)
    step_count: int = Field(default=0)
    max_steps: int = Field(default=24)
    score_so_far: float = Field(default=0.0)
    action_log: list[dict[str, Any]] = Field(default_factory=list)
