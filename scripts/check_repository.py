"""Reject dataset/output files in the index or reachable Git history; stdlib only."""

import argparse
import subprocess
import sys
from pathlib import PurePosixPath

BLOCKED_DIRECTORIES = {"dataset", "kaoti", "outputs", ".venv"}
BLOCKED_SUFFIXES = (".bmp", ".zip", ".7z", ".tar", ".tar.gz", ".tgz")


def forbidden(path: str) -> bool:
    normalized = path.lower()
    return bool(BLOCKED_DIRECTORIES.intersection(PurePosixPath(normalized).parts)) or (
        normalized.endswith(BLOCKED_SUFFIXES)
    )


def git(*arguments: str) -> bytes:
    return subprocess.run(["git", *arguments], check=True, capture_output=True).stdout


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", action="store_true")
    args = parser.parse_args()
    paths = set(git("ls-files", "-z").split(b"\0"))
    if args.history:
        for revision in git("rev-list", "--all").decode().splitlines():
            paths.update(git("ls-tree", "-r", "--name-only", "-z", revision).split(b"\0"))
    blocked = sorted(p.decode(errors="replace") for p in paths if p and forbidden(p.decode()))
    if blocked:
        print(
            "Blocked: datasets, generated outputs, environments or archives are tracked:",
            file=sys.stderr,
        )
        print("\n".join(blocked), file=sys.stderr)
        raise SystemExit(1)
    print("Repository data policy passed.")


if __name__ == "__main__":
    main()
