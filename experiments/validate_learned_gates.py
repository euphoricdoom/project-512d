"""Validate Phase 2 learned-gate success criteria."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def validate_results(results: dict) -> tuple[bool, list[str]]:
    """Return validation pass/fail and human-readable lines."""
    lines = []
    ok = True
    for config in results.get("configs", []):
        kernels = config["num_kernels"]
        for task in config["tasks"]:
            acc = float(task.get("accuracy", 0.0))
            forgetting = float(task.get("forgetting", 1.0))
            passed = acc >= 0.98 and forgetting < 0.01
            ok = ok and passed
            lines.append(
                f"{kernels}K {task['task_id']}: acc={acc:.1%}, forgetting={forgetting:.4%}, pass={passed}"
            )
    novel = results.get("novel_tasks", [])
    novel_passes = [float(task.get("accuracy", 0.0)) > 0.9 for task in novel]
    generalization = len(novel) >= 2 and all(novel_passes[:2])
    ok = ok and generalization
    lines.append(f"Novel-task gate: {len(novel)} tasks present, pass={generalization}")
    return ok, lines


def run_tests() -> bool:
    """Run learned-gate test subset."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_learned_gates.py",
            "tests/test_adaptive_helix.py",
            "-q",
        ],
        text=True,
        capture_output=True,
    )
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    return result.returncode == 0


def main() -> int:
    path = Path("outputs/learned_gates_benchmarks.json")
    if not path.exists():
        print("Missing outputs/learned_gates_benchmarks.json")
        return 1
    results = json.loads(path.read_text(encoding="utf-8"))
    results_ok, lines = validate_results(results)
    tests_ok = run_tests()
    report = ["# Phase 2 Validation Report", ""]
    report.extend(lines)
    report.append("")
    report.append(f"Tests pass: {tests_ok}")
    report.append(f"Overall pass: {results_ok and tests_ok}")
    out = Path("outputs/phase2_validation.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))
    return 0 if results_ok and tests_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
