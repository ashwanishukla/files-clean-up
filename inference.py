#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import json
import os
import re
from typing import Any

from openai import OpenAI

from client import FilesCleanUpEnv
from models import FilesCleanUpAction
from safe_cleanup_env.baseline import choose_destination
from safe_cleanup_env.sample_data import TASKS, TASK_LOOKUP


API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "openai/gpt-oss-120b:novita")
HF_TOKEN = os.getenv("HF_TOKEN")

# Optional - if you use from_docker_image():
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

SPACE_REPO_ID = "ashwanishukla24/files-clean-up"
MAX_OUTPUT_TOKENS = 220
TEMPERATURE = 0.0

SYSTEM_PROMPT = """
You are a safety-first computer file cleanup assistant.
Decide whether a single loose file should be moved into a personal folder or skipped.
Return strict JSON only with this schema:
{"decision":"move|skip","confidence":"high|medium|low","reason":"short reason"}

Rules:
- Prefer skip over risky or ambiguous actions.
- Skip anything that looks client-related, project-related, archived, bundled, or work-related.
- Move only clearly personal files.
- Do not include markdown or extra text.
""".strip()


def emit_log(tag: str, payload: dict[str, Any]) -> None:
    print(f"{tag} {json.dumps(payload, sort_keys=True, ensure_ascii=True)}", flush=True)


def extract_json_object(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def build_file_prompt(task_id: str, file_record: Any, allowed_destinations: list[str], protected_roots: list[str]) -> str:
    return json.dumps(
        {
            "task_id": task_id,
            "file": {
                "file_id": file_record["file_id"],
                "name": file_record["name"],
                "current_path": file_record["current_path"],
                "extension": file_record["extension"],
                "size_mb": file_record["size_mb"],
                "content_hint": file_record["content_hint"],
                "risk_flags": file_record["risk_flags"],
            },
            "allowed_destinations": allowed_destinations,
            "protected_roots": protected_roots,
            "instruction": "Return a move/skip decision for this file.",
        },
        sort_keys=True,
    )


def fallback_decision(file_record: dict[str, Any]) -> dict[str, str]:
    destination = choose_destination(type("FileLike", (), file_record))
    if destination:
        return {
            "decision": "move",
            "confidence": "medium",
            "reason": "deterministic fallback selected a personal destination",
        }
    return {
        "decision": "skip",
        "confidence": "high",
        "reason": "deterministic fallback judged the file risky or ambiguous",
    }


def normalize_plan(
    file_record: dict[str, Any],
    allowed_destinations: list[str],
    model_decision: dict[str, Any],
) -> list[FilesCleanUpAction]:
    risk_flags = set(file_record.get("risk_flags", []))
    reason = str(model_decision.get("reason", "")).strip() or "safety-first skip"
    heuristic_destination = choose_destination(type("FileLike", (), file_record))
    decision = str(model_decision.get("decision", "skip")).lower()
    confidence = str(model_decision.get("confidence", "low")).lower()

    actions = [FilesCleanUpAction(action_type="inspect_file", file_id=file_record["file_id"])]

    hard_skip_flags = {"project_like", "client_like", "archive_bundle", "ambiguous", "protected_root"}
    if risk_flags & hard_skip_flags:
        actions.append(
            FilesCleanUpAction(
                action_type="skip_file",
                file_id=file_record["file_id"],
                reason=reason or "risky file pattern",
            )
        )
        return actions

    if not heuristic_destination or heuristic_destination not in allowed_destinations:
        actions.append(
            FilesCleanUpAction(
                action_type="skip_file",
                file_id=file_record["file_id"],
                reason=reason or "no safe destination available",
            )
        )
        return actions

    if decision == "move" and confidence in {"high", "medium"}:
        actions.append(
            FilesCleanUpAction(
                action_type="propose_destination",
                file_id=file_record["file_id"],
                destination=heuristic_destination,
            )
        )
        actions.append(
            FilesCleanUpAction(
                action_type="move_file",
                file_id=file_record["file_id"],
                destination=heuristic_destination,
            )
        )
        return actions

    actions.append(
        FilesCleanUpAction(
            action_type="skip_file",
            file_id=file_record["file_id"],
            reason=reason,
        )
    )
    return actions


def decide_file(client: OpenAI, task_id: str, observation: Any, file_record: dict[str, Any]) -> dict[str, Any]:
    prompt = build_file_prompt(
        task_id=task_id,
        file_record=file_record,
        allowed_destinations=list(observation.allowed_destinations),
        protected_roots=list(observation.protected_roots),
    )
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=MAX_OUTPUT_TOKENS,
            temperature=TEMPERATURE,
        )
        content = (response.choices[0].message.content or "").strip()
        return extract_json_object(content)
    except Exception:
        return fallback_decision(file_record)


async def create_env() -> FilesCleanUpEnv:
    if LOCAL_IMAGE_NAME:
        return await FilesCleanUpEnv.from_docker_image(LOCAL_IMAGE_NAME)
    return await FilesCleanUpEnv.from_env(SPACE_REPO_ID)


async def run_inference() -> dict[str, float]:
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN must be set before running inference.py")

    llm_client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    env = await create_env()
    scores: dict[str, float] = {}
    total_steps = 0

    emit_log(
        "START",
        {
            "api_base_url": API_BASE_URL,
            "environment": SPACE_REPO_ID,
            "local_image_name": LOCAL_IMAGE_NAME,
            "model_name": MODEL_NAME,
            "task_count": len(TASKS),
        },
    )

    try:
        for task in TASKS:
            reset_result = await env.reset(task_id=task.task_id)
            observation = reset_result.observation

            for file_record in observation.candidate_files:
                model_decision = decide_file(llm_client, task.task_id, observation, file_record)
                planned_actions = normalize_plan(
                    file_record=file_record,
                    allowed_destinations=list(observation.allowed_destinations),
                    model_decision=model_decision,
                )

                for action in planned_actions:
                    result = await env.step(action)
                    observation = result.observation
                    total_steps += 1
                    emit_log(
                        "STEP",
                        {
                            "action_type": action.action_type,
                            "destination": action.destination,
                            "done": result.done,
                            "feedback": observation.feedback,
                            "file_id": action.file_id,
                            "reason": action.reason,
                            "reward": result.reward,
                            "step_count": total_steps,
                            "task_id": task.task_id,
                        },
                    )

            submit_result = await env.step(FilesCleanUpAction(action_type="submit"))
            total_steps += 1
            scores[task.task_id] = round(
                float(submit_result.observation.score_so_far),
                4,
            )
            emit_log(
                "STEP",
                {
                    "action_type": "submit",
                    "done": submit_result.done,
                    "feedback": submit_result.observation.feedback,
                    "reward": submit_result.reward,
                    "step_count": total_steps,
                    "task_id": task.task_id,
                },
            )
    finally:
        await env.close()

    emit_log(
        "END",
        {
            "model_name": MODEL_NAME,
            "scores": scores,
            "tasks": [task.task_id for task in TASKS],
            "total_steps": total_steps,
        },
    )
    return scores


def run_inference_sync() -> dict[str, float]:
    return asyncio.run(run_inference())


def main() -> None:
    run_inference_sync()


if __name__ == "__main__":
    main()
