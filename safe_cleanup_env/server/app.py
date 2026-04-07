from __future__ import annotations

from ..baseline import run_heuristic_baseline
from ..grader import grade_snapshot
from ..models import CleanupStateSnapshot
from ..sample_data import TASKS
from .environment import OpenEnvSafeCleanupEnvironment

try:
    from fastapi import FastAPI
except ImportError:  # pragma: no cover - optional runtime path
    FastAPI = None

def build_tasks_payload() -> dict:
    return {
        "tasks": [
            {
                "task_id": task.task_id,
                "title": task.title,
                "difficulty": task.difficulty,
                "allowed_destinations": task.allowed_destinations,
                "action_schema": {
                    "action_type": [
                        "inspect_file",
                        "propose_destination",
                        "move_file",
                        "skip_file",
                        "submit",
                    ],
                    "file_id": "string | null",
                    "destination": "string | null",
                    "reason": "string | null",
                },
            }
            for task in TASKS
        ]
    }


def build_grader_payload(task_id: str, snapshot_payload: dict) -> dict:
    from ..sample_data import TASK_LOOKUP

    task = TASK_LOOKUP[task_id]
    snapshot = CleanupStateSnapshot(
        proposed_destinations=dict(snapshot_payload.get("proposed_destinations", {})),
        moved_files=dict(snapshot_payload.get("moved_files", {})),
        skipped_files=dict(snapshot_payload.get("skipped_files", {})),
        inspected_files=list(snapshot_payload.get("inspected_files", [])),
        safety_violations=list(snapshot_payload.get("safety_violations", [])),
    )
    return grade_snapshot(task, snapshot).to_dict()


if FastAPI is not None:
    app = FastAPI(title="Safe Cleanup Agent Environment")
    environment = OpenEnvSafeCleanupEnvironment()

    @app.get("/health")
    def health_endpoint():
        return {"status": "healthy"}

    @app.post("/reset")
    def reset_endpoint(payload: dict | None = None):
        payload = payload or {}
        return environment.reset(**payload)

    @app.post("/step")
    def step_endpoint(payload: dict):
        return environment.step(payload)

    @app.get("/state")
    def state_endpoint():
        return environment.state

    @app.get("/tasks")
    def tasks_endpoint():
        return build_tasks_payload()

    @app.get("/baseline")
    def baseline_endpoint():
        return run_heuristic_baseline()

    @app.post("/grader")
    def grader_endpoint(payload: dict):
        return build_grader_payload(payload["task_id"], payload["snapshot"])
else:  # pragma: no cover - optional runtime path
    app = None
