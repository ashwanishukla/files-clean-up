from __future__ import annotations

from .environment import SafeCleanupEnvironment
from .sample_data import TASKS, TASK_LOOKUP


def choose_destination(file_record) -> str | None:
    name = file_record.name.lower()
    hint = file_record.content_hint.lower()
    flags = set(file_record.risk_flags)
    ext = file_record.extension.lower()

    if {"project_like", "client_like", "archive_bundle", "ambiguous"} & flags:
        return None
    if "screenshot" in flags:
        return "Pictures/Screenshots"
    if "screen_recording" in flags:
        return "Videos/Screen Recordings"
    if "medical" in hint:
        return "Documents/Medical"
    if "invoice" in name or "receipt" in name or "budget" in name:
        return "Documents/Finance"
    if "resume" in name:
        return "Documents/Career"
    if "passport" in name:
        return "Documents/Scans"
    if ext in {".jpg", ".jpeg", ".png", ".heic"}:
        return "Pictures/Family" if "family" in name else "Pictures/Photos"
    if ext in {".mp4", ".mov"}:
        return "Videos/Screen Recordings" if "screenrecording" in name.lower() else "Videos/Personal"
    if ext == ".mkv":
        return "Movies/Watch Later"
    if ext in {".txt", ".md"}:
        return "Documents/Notes"
    if ext in {".pdf", ".docx"}:
        return "Documents/Personal"
    if ext == ".xlsx":
        return "Documents/Finance"
    return None


def run_heuristic_baseline() -> dict[str, float]:
    env = SafeCleanupEnvironment()
    scores: dict[str, float] = {}

    for task in TASKS:
        env.reset(task.task_id)
        for file_record in task.candidate_files:
            env.step({"action_type": "inspect_file", "file_id": file_record.file_id})
            destination = choose_destination(file_record)
            if destination and destination in task.allowed_destinations:
                env.step(
                    {
                        "action_type": "propose_destination",
                        "file_id": file_record.file_id,
                        "destination": destination,
                    }
                )
                result = env.step(
                    {
                        "action_type": "move_file",
                        "file_id": file_record.file_id,
                        "destination": destination,
                    }
                )
            else:
                result = env.step(
                    {
                        "action_type": "skip_file",
                        "file_id": file_record.file_id,
                        "reason": "baseline judged the file too risky or unclear",
                    }
                )

        if not result["done"]:
            result = env.step({"action_type": "submit"})
        scores[task.task_id] = result["info"]["grading"]["total"]

    return scores
