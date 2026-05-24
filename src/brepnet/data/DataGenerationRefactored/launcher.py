from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable


def project_root() -> Path:
    return Path(__file__).resolve().parent


def run_checked(cmd: list[str], env: dict[str, str] | None = None, log_path: Path | None = None) -> None:
    print(format_command(cmd))
    if log_path is None:
        subprocess.run(cmd, check=True, env=env)
        return

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        subprocess.run(cmd, check=True, env=env, stdout=log_file, stderr=subprocess.STDOUT)


def run_parallel_ranked(
    commands: Iterable[tuple[int, list[str], dict[str, str], Path]],
) -> None:
    processes: list[tuple[int, Path, subprocess.Popen[bytes]]] = []
    for rank, cmd, env, log_path in commands:
        print(f"rank={rank} log={log_path}")
        print(format_command(cmd))
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = log_path.open("wb")
        try:
            proc = subprocess.Popen(cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT)
        finally:
            log_file.close()
        processes.append((rank, log_path, proc))

    failed: list[tuple[int, int, Path]] = []
    for rank, log_path, proc in processes:
        code = proc.wait()
        if code != 0:
            failed.append((rank, code, log_path))

    if failed:
        for rank, code, log_path in failed:
            print(f"rank={rank} failed exit_code={code} log={log_path}", file=sys.stderr)
        raise SystemExit(1)


def rank_env(rank: int) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["CUDA_VISIBLE_DEVICES"] = str(rank)
    env["RANK"] = str(rank)
    return env


def format_command(cmd: list[str]) -> str:
    return " ".join(str(part) for part in cmd)

