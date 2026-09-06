"""Reject dataset/output files in the index or reachable Git history; stdlib only."""

import argparse
import subprocess
import sys
from fnmatch import fnmatchcase
from pathlib import PurePosixPath

BLOCKED_DIRECTORIES = {"dataset", "kaoti", "outputs", "output", "tmp", ".venv"}
BLOCKED_SUFFIXES = (".bmp", ".zip", ".7z", ".tar", ".tar.gz", ".tgz", ".rar")
WEIGHT_DIRECTORIES = {"weights", "checkpoints"}
WEIGHT_SUFFIXES = (".pt", ".pth", ".ckpt", ".safetensors", ".onnx", ".pt.tmp", ".pth.tmp")
LOCAL_ONLY = (
    "run_experiments.py",
    "src/calibration_experiments.py",
    "src/experiment_*.py",
    "scripts/build_manual.py",
    "scripts/plot_c_comparison.py",
    "tests/test_experiment_design.py",
    "tests/test_manual.py",
    ".vscode/*",
)
DELIVERY_FILES = {
    "delivery/calibration_results.csv",
    "delivery/deeplearning_results.csv",
    "delivery/pose_results.csv",
    "delivery/technical_manual.html",
    "delivery/technical_manual.pdf",
}


def forbidden(path: str, *, historical: bool = False) -> bool:
    """忽略大小写，判断 Git 路径是否命中禁用目录或文件后缀。"""
    normalized = path.lower()
    if WEIGHT_DIRECTORIES.intersection(PurePosixPath(normalized).parts) or normalized.endswith(
        (*WEIGHT_SUFFIXES, ".tmp")
    ):
        return True
    if not historical and normalized.startswith("delivery/"):
        return normalized not in DELIVERY_FILES
    if not historical and any(fnmatchcase(normalized, pattern) for pattern in LOCAL_ONLY):
        return True
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
    blocked = {p.decode(errors="replace") for p in paths if p and forbidden(p.decode())}
    if args.history:
        for revision in git("rev-list", "--all").decode().splitlines():
            historical_paths = git("ls-tree", "-r", "--name-only", "-z", revision).split(b"\0")
            blocked.update(
                p.decode(errors="replace")
                for p in historical_paths
                if p and forbidden(p.decode(), historical=True)
            )
    if blocked:
        print(
            "Blocked: local experiments, datasets, outputs or unapproved artifacts are tracked:",
            file=sys.stderr,
        )
        print("\n".join(sorted(blocked)), file=sys.stderr)
        raise SystemExit(1)
    print("Repository data policy passed.")


if __name__ == "__main__":
    main()
