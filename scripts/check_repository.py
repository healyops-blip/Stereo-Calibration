"""Reject dataset/output files in the index or reachable Git history; stdlib only."""

import argparse
import subprocess
import sys
from pathlib import PurePosixPath

BLOCKED_DIRECTORIES = {"dataset", "kaoti", "outputs", ".venv"}
BLOCKED_SUFFIXES = (".bmp", ".zip", ".7z", ".tar", ".tar.gz", ".tgz")


def forbidden(path: str) -> bool:
    """忽略大小写，判断 Git 路径是否命中禁用目录或文件后缀。"""
    normalized = path.lower()
    return bool(BLOCKED_DIRECTORIES.intersection(PurePosixPath(normalized).parts)) or (
        normalized.endswith(BLOCKED_SUFFIXES)
    )


def git(*arguments: str) -> bytes:
    """运行传入 arguments 的 Git 命令并返回标准输出字节。

    Raises:
        subprocess.CalledProcessError: Git 返回非零状态。
    """
    return subprocess.run(["git", *arguments], check=True, capture_output=True).stdout


def main() -> None:
    """检查已跟踪文件；传 --history 时同时检查所有可达历史树。

    不改 Git 索引或提交；发现数据、产物或环境文件时打印路径并以状态 1 退出。
    """
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
