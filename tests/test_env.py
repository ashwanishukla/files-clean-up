from __future__ import annotations

import unittest

from safe_cleanup_env.baseline import run_heuristic_baseline
from safe_cleanup_env.client import SafeCleanupAction, SafeCleanupEnvClient
from safe_cleanup_env.environment import SafeCleanupEnvironment, build_reference_plan
from safe_cleanup_env.sample_data import TASKS


class SafeCleanupEnvironmentTests(unittest.TestCase):
    def test_reference_plan_scores_high(self) -> None:
        env = SafeCleanupEnvironment()

        for task in TASKS:
            env.reset(task.task_id)
            plan = build_reference_plan(task)
            for action in plan:
                result = env.step(action)
            self.assertTrue(result["done"])
            self.assertGreaterEqual(result["info"]["grading"]["total"], 0.95)

    def test_protected_or_risky_file_move_is_penalized(self) -> None:
        env = SafeCleanupEnvironment()
        env.reset("downloads_sort_medium")
        result = env.step(
            {
                "action_type": "move_file",
                "file_id": "archive-zip-1",
                "destination": "Documents/Archives",
            }
        )
        self.assertLess(result["reward"], 0)

    def test_typed_client_wraps_environment(self) -> None:
        client = SafeCleanupEnvClient()
        observation = client.reset("desktop_cleanup_easy")
        self.assertEqual(observation.task_id, "desktop_cleanup_easy")

        result = client.step(
            SafeCleanupAction(
                action_type="inspect_file",
                file_id="desktop-photo-1",
            )
        )
        self.assertGreaterEqual(result.reward, 0)

    def test_baseline_scores_are_reproducible(self) -> None:
        first = run_heuristic_baseline()
        second = run_heuristic_baseline()
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
