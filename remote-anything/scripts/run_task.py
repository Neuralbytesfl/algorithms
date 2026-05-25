#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]

TASK_DIR = ROOT / "remote-anything" / "tasks"
RESULT_DIR = ROOT / "remote-anything" / "results"
WORKSPACE_DIR = ROOT / "remote-anything" / "workspaces"
MEMORY_FILE = ROOT / "remote-anything" / "memory" / "index.jsonl"

ALLOWED_TASK_TYPES = {
    "benchmark",
    "repo_analysis",
    "code_generation",
    "dataset_generation",
    "memory_update",
}

DEFAULT_POLICY = {
    "allow_shell": False,
    "allow_network": False,
    "allow_subprocess": False,
    "allow_file_delete": False,
    "allow_outside_workspace_write": False,
}

BLOCKED_IMPORTS = {
    "socket",
    "requests",
    "urllib",
    "ftplib",
    "telnetlib",
}

WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)
MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            h.update(chunk)

    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_task(task: dict[str, Any]) -> list[str]:
    errors = []

    required = [
        "task_id",
        "task_type",
        "executor",
        "inputs",
        "constraints",
        "safety",
    ]

    for field in required:
        if field not in task:
            errors.append(f"missing field: {field}")

    if task.get("task_type") not in ALLOWED_TASK_TYPES:
        errors.append("invalid task type")

    safety = task.get("safety", {})

    for key, value in DEFAULT_POLICY.items():
        if safety.get(key, value) != value:
            errors.append(f"policy violation: {key}")

    return errors


def static_validate_script(script_path: Path) -> list[str]:
    errors = []

    if not script_path.exists():
        errors.append("script missing")
        return errors

    text = script_path.read_text(encoding="utf-8")

    for blocked in BLOCKED_IMPORTS:
        if f"import {blocked}" in text:
            errors.append(f"blocked import detected: {blocked}")

    dangerous_patterns = [
        "os.system(",
        "subprocess.",
        "shutil.rmtree(",
    ]

    for pattern in dangerous_patterns:
        if pattern in text:
            errors.append(f"dangerous pattern detected: {pattern}")

    return errors


def create_workspace(task_id: str) -> Path:
    path = WORKSPACE_DIR / task_id

    if path.exists():
        shutil.rmtree(path)

    path.mkdir(parents=True)

    return path


def append_memory(event: dict[str, Any]) -> None:
    previous_hash = ""

    if MEMORY_FILE.exists():
        lines = MEMORY_FILE.read_text().splitlines()

        if lines:
            try:
                last = json.loads(lines[-1])
                previous_hash = last.get("entry_hash", "")
            except Exception:
                previous_hash = ""

    payload = {
        "timestamp": utc_now(),
        "previous_hash": previous_hash,
        **event,
    }

    encoded = json.dumps(payload, sort_keys=True)
    payload["entry_hash"] = sha256_text(encoded)

    with open(MEMORY_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


def execute_python(script: Path, workspace: Path) -> dict[str, Any]:
    start = time.time()

    proc = subprocess.run(
        [sys.executable, str(script)],
        cwd=workspace,
        capture_output=True,
        text=True,
        timeout=60,
    )

    elapsed = time.time() - start

    return {
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "elapsed_seconds": elapsed,
    }


def save_result(task_id: str, result: dict[str, Any]) -> Path:
    path = RESULT_DIR / f"{task_id}.result.json"

    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    return path


def run_task(task_path: Path) -> int:
    task = load_json(task_path)

    validation_errors = validate_task(task)

    if validation_errors:
        print("TASK VALIDATION FAILED")

        for err in validation_errors:
            print("-", err)

        return 1

    script_path = ROOT / task["inputs"]["script"]

    static_errors = static_validate_script(script_path)

    if static_errors:
        print("STATIC VALIDATION FAILED")

        for err in static_errors:
            print("-", err)

        return 1

    workspace = create_workspace(task["task_id"])

    result = execute_python(script_path, workspace)

    result_object = {
        "task_id": task["task_id"],
        "completed_at": utc_now(),
        "status": (
            "success"
            if result["returncode"] == 0
            else "failed"
        ),
        "metrics": {
            "elapsed_seconds": result["elapsed_seconds"],
        },
        "stdout": result["stdout"],
        "stderr": result["stderr"],
        "artifacts": [],
    }

    result_path = save_result(
        task["task_id"],
        result_object,
    )

    result_hash = sha256_file(result_path)

    append_memory({
        "event": "task_completed",
        "task_id": task["task_id"],
        "result_hash": result_hash,
    })

    print(json.dumps(result_object, indent=2))

    return 0


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: run_task.py <task.json>")
        return 1

    return run_task(Path(sys.argv[1]))


if __name__ == "__main__":
    raise SystemExit(main())
