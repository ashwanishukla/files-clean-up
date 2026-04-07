---
title: Files Clean Up
sdk: docker
app_port: 8000
tags:
  - openenv
  - reinforcement-learning
  - agents
  - file-organization
---

# Safe Cleanup Agent Environment

This repository contains a mini RL environment for a safety-critical real-world task:

**clean up loose personal files across a computer without ever disturbing protected project, client, or work files**

The environment models the problem as an agent that sees a snapshot of candidate files, protected roots, and allowed destination folders. The agent must inspect files, decide whether they are safe to move, route them into the right organized folder, and skip anything risky or ambiguous.

## Why this problem matters

People accumulate files on Desktop, Downloads, and Documents:

- screenshots
- photos
- videos
- receipts
- scans
- notes
- movie clips

The hard part is not organizing obvious files. The hard part is doing it **safely**:

- do not touch project code
- do not touch client deliverables
- do not move archives or ambiguous bundles blindly
- do not rename or disturb protected files by mistake

That makes this a strong agent task: the reward function must value both productivity and restraint.

## Environment summary

Each task provides:

- loose candidate files that may need cleanup
- protected roots that must never be disturbed
- allowed destination folders
- deterministic ground-truth expectations for each candidate file

The agent can:

- inspect a file
- propose a destination
- move a file
- skip a file as unsafe/ambiguous
- submit the episode

The grader scores:

- correct final decisions
- safety behavior
- completion
- avoidance of protected-file mistakes

## Tasks

The environment ships with 3 tasks and explicit difficulty progression:

1. `desktop_cleanup_easy`
   Straightforward personal media and documents with obvious destinations.
2. `downloads_sort_medium`
   Mixed files with one risky archive that should be skipped.
3. `safety_first_hard`
   Harder distinctions between personal documents and files that look like client/project/work artifacts.

## Project structure

```text
safe_cleanup_env/
  __init__.py
  baseline.py
  cli.py
  client.py
  environment.py
  grader.py
  models.py
  sample_data.py
  server/
    __init__.py
    app.py
    environment.py
    Dockerfile
tests/
  test_env.py
```

## Action space

Actions are JSON-like dictionaries.

Inspect a file:

```python
{"action_type": "inspect_file", "file_id": "desktop-photo-1"}
```

Propose a destination:

```python
{"action_type": "propose_destination", "file_id": "desktop-photo-1", "destination": "Pictures/Photos"}
```

Move a file:

```python
{"action_type": "move_file", "file_id": "desktop-photo-1", "destination": "Pictures/Photos"}
```

Skip a file:

```python
{"action_type": "skip_file", "file_id": "archive-zip-1", "reason": "archive bundle is too risky to auto-move"}
```

Submit:

```python
{"action_type": "submit"}
```

## Observation space

Observations contain:

- task metadata
- candidate file metadata
- protected roots
- allowed destinations
- unresolved file ids
- action history
- feedback and current score

## Local usage

List tasks:

```bash
python3 -m safe_cleanup_env.cli list-tasks
```

Show one task:

```bash
python3 -m safe_cleanup_env.cli show-task desktop_cleanup_easy
```

Run the deterministic heuristic baseline:

```bash
python3 -m safe_cleanup_env.cli baseline
```

Run tests:

```bash
python3 -m unittest discover -s tests -v
```

## Baseline scores

The included baseline is a deterministic safety-first heuristic policy.

- `desktop_cleanup_easy`: 0.8000
- `downloads_sort_medium`: 1.0000
- `safety_first_hard`: 0.9000

These are reproducible because the baseline does not use randomness.

## API surface

Core environment:

- `reset(task_id: str | None = None) -> dict`
- `step(action: dict) -> dict`
- `state() -> dict`
- `available_tasks() -> list[str]`

Server helper endpoints:

- `GET /tasks`
- `GET /baseline`
- `POST /grader`

## OpenEnv packaging

The repository includes:

- `openenv.yaml`
- an environment adapter under `safe_cleanup_env/server/environment.py`
- a server app under `safe_cleanup_env/server/app.py`
- a root `Dockerfile` for Hugging Face Spaces
- a package Dockerfile under `safe_cleanup_env/server/Dockerfile`

Optional server/runtime dependencies are listed in `requirements-openenv.txt`.
