from __future__ import annotations

import argparse
import json

from .baseline import run_heuristic_baseline
from .sample_data import TASKS, TASK_LOOKUP


def _list_tasks() -> int:
    for task in TASKS:
        print(f"{task.task_id} :: {task.difficulty} :: {task.title}")
    return 0


def _show_task(task_id: str) -> int:
    task = TASK_LOOKUP[task_id]
    print(json.dumps(task.to_observation(), indent=2))
    return 0


def _run_baseline() -> int:
    scores = run_heuristic_baseline()
    print(json.dumps(scores, indent=2, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Safe cleanup agent environment")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("list-tasks", help="List available tasks.")
    subparsers.add_parser("baseline", help="Run the deterministic heuristic baseline.")
    show_parser = subparsers.add_parser("show-task", help="Show one task as JSON.")
    show_parser.add_argument("task_id")

    args = parser.parse_args()
    if args.command == "list-tasks":
        return _list_tasks()
    if args.command == "baseline":
        return _run_baseline()
    if args.command == "show-task":
        return _show_task(args.task_id)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
