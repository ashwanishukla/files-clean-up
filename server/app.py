from __future__ import annotations

try:
    from openenv.core.env_server.http_server import create_app
except Exception as exc:  # pragma: no cover
    raise ImportError("openenv-core[core] is required to serve this environment") from exc

from models import FilesCleanUpAction, FilesCleanUpObservation
from safe_cleanup_env.grader import grade_snapshot
from safe_cleanup_env.models import CleanupStateSnapshot
from safe_cleanup_env.sample_data import TASKS, TASK_LOOKUP
from server.environment import FilesCleanUpEnvironment


app = create_app(
    FilesCleanUpEnvironment,
    FilesCleanUpAction,
    FilesCleanUpObservation,
    env_name="files_clean_up",
    max_concurrent_envs=8,
)


@app.get("/tasks")
def tasks_endpoint():
    return {
        "tasks": [
            {
                "task_id": task.task_id,
                "title": task.title,
                "difficulty": task.difficulty,
                "allowed_destinations": task.allowed_destinations,
                "action_schema": FilesCleanUpAction.model_json_schema(),
            }
            for task in TASKS
        ]
    }


@app.get("/baseline")
def baseline_endpoint():
    from inference import run_inference_sync

    return run_inference_sync()


@app.post("/grader")
def grader_endpoint(payload: dict):
    task = TASK_LOOKUP[payload["task_id"]]
    snapshot_payload = payload["snapshot"]
    snapshot = CleanupStateSnapshot(
        proposed_destinations=dict(snapshot_payload.get("proposed_destinations", {})),
        moved_files=dict(snapshot_payload.get("moved_files", {})),
        skipped_files=dict(snapshot_payload.get("skipped_files", {})),
        inspected_files=list(snapshot_payload.get("inspected_files", [])),
        safety_violations=list(snapshot_payload.get("safety_violations", [])),
    )
    return grade_snapshot(task, snapshot).to_dict()


def main(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
