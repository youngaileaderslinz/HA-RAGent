from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


CONTAINER_NAME = "homeassistant"
CONTAINER_TESTS_PATH = "/config/tests"
REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
HOST_TESTS_PATH = REPOSITORY_ROOT / "tests"

def run(*command: str) -> None:
    """Run a Docker command and return its exit status to the caller."""
    completed = subprocess.run(command, check=False)
    if completed.returncode:
        raise SystemExit(completed.returncode)

def verify_container_is_running() -> None:
    """Verify that the Home Assistant Docker container is running."""
    status = subprocess.run(
        ["docker", "inspect", "--format", "{{.State.Running}}", CONTAINER_NAME],
        check=False,
        capture_output=True,
        text=True,
    )
    is_running = status.returncode == 0 and status.stdout.strip().lower() == "true"

    if not is_running:
        raise SystemExit("The 'homeassistant' container is not running. Start it with: docker compose -f dev/docker-compose.yaml up -d homeassistant")

def setup_environment() -> None:
    """Clean old tests in the container and prepare the environment."""
    run("docker", "exec", CONTAINER_NAME, "sh", "-c", f"rm -rf {CONTAINER_TESTS_PATH} && mkdir -p {CONTAINER_TESTS_PATH}")

    with tempfile.TemporaryDirectory(prefix="ha-ragent-tests-") as temporary_directory:
        staged_tests_path = Path(temporary_directory) / "tests"
        shutil.copytree(HOST_TESTS_PATH, staged_tests_path, ignore=shutil.ignore_patterns(".pytest_cache", "__pycache__", "*.pyc"))
        run("docker", "cp", str(staged_tests_path), f"{CONTAINER_NAME}:/config/")

def run_tests_in_container() -> None:
    """Run pytest inside the Home Assistant Docker container."""
    run("docker", "exec", CONTAINER_NAME, "python", "-m", "pytest", "--color=yes", CONTAINER_TESTS_PATH, ' '.join(sys.argv[1:]))

if __name__ == "__main__":
    verify_container_is_running()
    setup_environment()
    run_tests_in_container()