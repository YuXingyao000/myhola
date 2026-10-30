#!/usr/bin/env python3
"""Wait for, pull, and verify one immutable training run from the compute node."""

from __future__ import annotations

import argparse
import ctypes
import datetime as dt
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import signal
import stat
import subprocess
import sys
import time


READY = "metadata/READY_FOR_RETURN.json"
MANIFEST = "SHA256SUMS"
FILES = "RETURN_FILES.txt"
CONTROL_FILES = {READY, MANIFEST, FILES}
HASH_RE = re.compile(r"[0-9a-f]{64}\Z")


class RemoteReadError(RuntimeError):
    def __init__(self, code: int, message: str):
        super().__init__(message)
        self.code = code


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def log(message: str) -> None:
    print(f"{now()} {message}", flush=True)


def validate_relative_path(value: str) -> str:
    if not value or value.startswith("/") or "\\" in value:
        raise ValueError(f"Unsafe relative path: {value!r}")
    if any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise ValueError(f"Control character in path: {value!r}")
    if any(part in {"", ".", ".."} for part in value.split("/")):
        raise ValueError(f"Non-canonical relative path: {value!r}")
    return value


def validate_run_dir(value: str) -> str:
    if not value.startswith("/") or value == "/":
        raise ValueError("--run-dir must be an absolute, non-root remote path")
    validate_relative_path(value[1:])
    return value


def parse_files(data: bytes) -> list[str]:
    text = data.decode("utf-8")
    paths = [validate_relative_path(line) for line in text.splitlines()]
    if not paths or len(paths) != len(set(paths)):
        raise ValueError("RETURN_FILES.txt must be nonempty and contain no duplicates")
    if not CONTROL_FILES.issubset(paths):
        raise ValueError(f"RETURN_FILES.txt must include {sorted(CONTROL_FILES)}")
    return paths


def parse_manifest(data: bytes, files: list[str]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for line in data.decode("utf-8").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64}) [ *](.+)", line)
        if match is None:
            raise ValueError(f"Invalid SHA256SUMS line: {line!r}")
        digest, relative = match.groups()
        validate_relative_path(relative)
        if relative in hashes:
            raise ValueError(f"Duplicate SHA256SUMS path: {relative}")
        hashes[relative] = digest
    expected = set(files) - CONTROL_FILES
    if not expected or set(hashes) != expected:
        raise ValueError(
            "SHA256SUMS must cover exactly the payload files, excluding the three "
            f"control files; missing={sorted(expected - set(hashes))}, "
            f"unexpected={sorted(set(hashes) - expected)}"
        )
    return hashes


def digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_ready(data: bytes, run_dir: str) -> dict:
    value = json.loads(data)
    if not isinstance(value, dict):
        raise ValueError("READY must contain a JSON object")
    if value.get("status") not in {"completed", "failed", "interrupted"}:
        raise ValueError("READY.status must be completed, failed, or interrupted")
    if value.get("run_dir") != run_dir:
        raise ValueError("READY.run_dir does not match --run-dir")
    for name in ["manifest_sha256", "files_sha256"]:
        if not isinstance(value.get(name), str) or not HASH_RE.fullmatch(value[name]):
            raise ValueError(f"READY.{name} must contain a lowercase SHA256 digest")
    return value


def assert_no_symlinks(path: Path) -> None:
    for candidate in [path, *path.parents]:
        if candidate.is_symlink():
            raise ValueError(f"Symlink is forbidden: {candidate}")


def verify_local_tree(
    root: Path, files: list[str], hashes: dict[str, str],
    ready_bytes: bytes, manifest_bytes: bytes, files_bytes: bytes,
) -> dict:
    assert_no_symlinks(root)
    seen: set[str] = set()
    total_bytes = 0
    for directory, dirs, names in os.walk(root, followlinks=False):
        for name in dirs:
            path = Path(directory) / name
            if path.is_symlink():
                raise ValueError(f"Symlink directory is forbidden: {path}")
        for name in names:
            path = Path(directory) / name
            if not stat.S_ISREG(path.lstat().st_mode):
                raise ValueError(f"Non-regular file is forbidden: {path}")
            seen.add(path.relative_to(root).as_posix())
    if seen != set(files):
        raise ValueError(
            f"Staging file list mismatch: missing={sorted(set(files) - seen)}, "
            f"unexpected={sorted(seen - set(files))}"
        )
    controls = {READY: ready_bytes, MANIFEST: manifest_bytes, FILES: files_bytes}
    for relative in files:
        path = root / relative
        assert_no_symlinks(path)
        expected = digest_bytes(controls[relative]) if relative in controls else hashes[relative]
        actual = digest_file(path)
        if actual != expected:
            raise ValueError(f"SHA256 mismatch: {relative}; expected={expected}, actual={actual}")
        total_bytes += path.stat().st_size
    return {"file_count": len(files), "total_bytes": total_bytes}


def write_json(path: Path, value: dict) -> None:
    assert_no_symlinks(path)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_publish(staging: Path, destination: Path) -> None:
    """Linux renameat2 avoids the check/rename race that could replace an empty dir."""
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("Atomic no-overwrite publication requires Linux renameat2")
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    result = renameat2(-100, os.fsencode(staging), -100, os.fsencode(destination), 1)
    if result != 0:
        error = ctypes.get_errno()
        if error == errno.EEXIST:
            raise FileExistsError(f"Destination already exists; preserved: {destination}")
        raise OSError(error, os.strerror(error), str(destination))


def ssh_options(args: argparse.Namespace) -> list[str]:
    return [
        "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
        "-o", "ServerAliveInterval=20", "-o", "ServerAliveCountMax=3",
        "-p", str(args.port),
    ]


def remote(args: argparse.Namespace, command: str, script: bytes | None = None,
           timeout: int = 300) -> subprocess.CompletedProcess:
    return subprocess.run(
        [*ssh_options(args), "--", args.host, command], input=script,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout, check=False,
    )


def read_remote(args: argparse.Namespace, relative: str, allow_missing: bool = False) -> bytes | None:
    path = shlex.quote(args.run_dir + "/" + relative)
    command = (
        f"if test -L {path}; then exit 45; "
        f"elif test -f {path}; then cat -- {path}; else exit 44; fi"
    )
    result = remote(args, command)
    if result.returncode == 44 and allow_missing:
        return None
    if result.returncode:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise RemoteReadError(result.returncode, f"SSH read {relative}: exit={result.returncode}; {detail}")
    return result.stdout


REMOTE_VERIFY = r"""set -eu
run=$1
manifest_hash=$2
files_hash=$3
ready_hash=$4
shift 4
check_path() {
    candidate=$1
    while test "$candidate" != /; do
        if test -L "$candidate"; then
            printf 'Symlink forbidden: %s\n' "$candidate" >&2
            exit 46
        fi
        candidate=${candidate%/*}
        test -n "$candidate" || candidate=/
    done
}
check_path "$run"
for relative do
    check_path "$run/$relative"
    test -f "$run/$relative" || exit 47
done
cd -- "$run"
printf '%s  %s\n' "$manifest_hash" SHA256SUMS "$files_hash" RETURN_FILES.txt "$ready_hash" metadata/READY_FOR_RETURN.json | sha256sum --check --strict -
sha256sum --check --strict SHA256SUMS
"""


def verify_remote(args: argparse.Namespace, files: list[str], ready: dict, ready_bytes: bytes) -> str:
    arguments = [args.run_dir, ready["manifest_sha256"], ready["files_sha256"], digest_bytes(ready_bytes), *files]
    result = remote(args, "sh -s -- " + shlex.join(arguments), REMOTE_VERIFY.encode(), timeout=86400)
    output = (result.stdout + result.stderr).decode("utf-8", errors="replace")
    if result.returncode:
        raise RuntimeError(f"Remote SHA256/path verification failed, exit={result.returncode}:\n{output}")
    return output


def execute(args: argparse.Namespace) -> int:
    validate_run_dir(args.run_dir)
    if not re.fullmatch(r"[A-Za-z0-9_.@:\[\]-]+", args.host) or args.host.startswith("-"):
        raise ValueError("--host must be an SSH host/alias, optionally prefixed by user@")
    if not 1 <= args.port <= 65535 or args.poll_seconds <= 0:
        raise ValueError("Invalid port or polling interval")
    validate_run_dir(args.destination_root)
    root = Path(args.destination_root)
    assert_no_symlinks(root)
    root.mkdir(parents=True, exist_ok=True)
    name = args.run_dir.rsplit("/", 1)[1]
    final = root / name
    staging = root / ("." + name + ".return-staging")
    state_path = root / ("." + name + ".return-state.json")
    receipt_path = root / (name + ".return_receipt.json")
    lock_path = root / ("." + name + ".return.lock")
    for path in [final, staging, state_path, receipt_path, lock_path]:
        assert_no_symlinks(path)
    retry = shlex.join([sys.executable, str(Path(__file__).absolute()), *sys.argv[1:]])
    identity = {"host": args.host, "port": args.port, "run_dir": args.run_dir, "destination": str(final)}
    with lock_path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another pull process owns {lock_path}") from exc
        if final.exists() or receipt_path.exists():
            raise FileExistsError(f"Final destination or receipt already exists; refusing overwrite: {final}")
        state = json.loads(state_path.read_text()) if state_path.exists() else {}
        if state and state.get("identity") != identity:
            raise ValueError(f"Existing transfer state belongs to another source: {state_path}")
        if staging.exists() and not state:
            raise ValueError(f"Unowned staging directory exists; preserved: {staging}")
        state.update(identity=identity, retry_command=retry, pid=os.getpid(), updated_at=now())

        def update(status: str, **values) -> None:
            state.update(status=status, updated_at=now(), **values)
            write_json(state_path, state)

        try:
            update("waiting")
            log(f"Waiting for {args.host}:{args.run_dir}/{READY}; status={state_path}")
            while True:
                try:
                    ready_bytes = read_remote(args, READY, allow_missing=True)
                except (RemoteReadError, subprocess.TimeoutExpired) as exc:
                    if isinstance(exc, RemoteReadError) and exc.code != 255:
                        raise
                    update("waiting_connection_error", last_error=str(exc))
                    log(f"Connection unavailable; retry in {args.poll_seconds}s: {exc}")
                    time.sleep(args.poll_seconds)
                    continue
                if ready_bytes is not None:
                    break
                time.sleep(args.poll_seconds)
            ready = parse_ready(ready_bytes, args.run_dir)
            manifest_bytes = read_remote(args, MANIFEST)
            files_bytes = read_remote(args, FILES)
            assert manifest_bytes is not None and files_bytes is not None
            if digest_bytes(manifest_bytes) != ready["manifest_sha256"]:
                raise ValueError("Remote SHA256SUMS differs from READY.manifest_sha256")
            if digest_bytes(files_bytes) != ready["files_sha256"]:
                raise ValueError("Remote RETURN_FILES.txt differs from READY.files_sha256")
            ready_digest = digest_bytes(ready_bytes)
            if state.get("ready_sha256") not in {None, ready_digest}:
                raise ValueError("READY changed since a previous attempt; preserving the previous staging data")
            files = parse_files(files_bytes)
            hashes = parse_manifest(manifest_bytes, files)
            update("verifying_source", ready_sha256=ready_digest, training_status=ready["status"])
            log(f"Remote run ended with status={ready['status']}; verifying {len(hashes)} payload files")
            verify_remote(args, files, ready, ready_bytes)
            staging.mkdir(exist_ok=True)
            assert_no_symlinks(staging)
            # Reject unexpected paths before rsync can write through a stale local link.
            for directory, dirs, names in os.walk(staging, followlinks=False):
                for entry in [*dirs, *names]:
                    path = Path(directory) / entry
                    assert_no_symlinks(path)
                    mode = path.lstat().st_mode
                    if not (stat.S_ISDIR(mode) or stat.S_ISREG(mode)):
                        raise ValueError(f"Non-regular staging entry: {path}")
            list_path = root / ("." + name + ".return-files.tmp")
            assert_no_symlinks(list_path)
            with list_path.open("wb") as stream:
                stream.write(files_bytes)
            command = [
                "rsync", "-rt", "--partial", "--checksum", "--protect-args",
                "--files-from=" + str(list_path), "-e", shlex.join(ssh_options(args)),
                "--", args.host + ":" + args.run_dir + "/", str(staging) + "/",
            ]
            update("transferring", staging=str(staging))
            log(f"Pulling into {staging}")
            subprocess.run(command, check=True)
            update("verifying_destination")
            statistics = verify_local_tree(staging, files, hashes, ready_bytes, manifest_bytes, files_bytes)
            receipt = {
                "status": "returned_and_verified", "verified_at": now(), "identity": identity,
                "training_status": ready["status"], "ready_sha256": ready_digest,
                "manifest_sha256": ready["manifest_sha256"], "files_sha256": ready["files_sha256"],
                "source_sha256_verified": True, "destination_sha256_verified": True, **statistics,
            }
            atomic_publish(staging, final)
            write_json(receipt_path, receipt)
            update("returned_and_verified", receipt=str(receipt_path), **statistics)
            log(f"Returned and verified: {final}; receipt={receipt_path}; training_status={ready['status']}")
            return 0
        except (Exception, KeyboardInterrupt) as exc:
            update("interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", last_error=str(exc))
            log(f"Pull did not complete: {exc}\nFiles preserved. Retry: {retry}")
            if final.exists():
                log(f"Final directory exists; inspect it and {state_path}; automatic overwrite is forbidden")
            return 130 if isinstance(exc, KeyboardInterrupt) else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Absolute remote run directory (no trailing slash)")
    parser.add_argument("--host", default="root@172.31.224.142")
    parser.add_argument("--port", default=12222, type=int)
    parser.add_argument("--destination-root", default="/mnt/d/python/checkpoints/uv_face_encoder/candidate")
    parser.add_argument("--poll-seconds", default=60, type=int)
    args = parser.parse_args()
    def interrupt(signum, frame):
        raise KeyboardInterrupt(f"Received signal {signum}")
    signal.signal(signal.SIGINT, interrupt)
    signal.signal(signal.SIGTERM, interrupt)
    try:
        return execute(args)
    except Exception as exc:
        log(f"Cannot start pull: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
