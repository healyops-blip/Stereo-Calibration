"""Run every development check and save an auditable report; exit nonzero on failure."""

import argparse
import subprocess
import sys
import time
from pathlib import Path

from src.export import write_json


def main() -> None:
    """执行本地静态检查、测试与依赖检查，并保存逐项退出码和日志。

    --with-results 额外调用正式 CSV 回读验证。写入
    outputs/quality/quality_checks.json；任一检查失败以状态 1 退出。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--with-results", action="store_true", help="Also verify existing CSV outputs"
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    targets = [
        "src",
        "scripts",
        "tests",
        "run_pipeline.py",
        "run_experiments.py",
        "visualize_calibration.py",
        "validate_pose_accuracy.py",
        "check_quality.py",
    ]
    commands = [
        ["-m", "ruff", "check", *targets],
        ["-m", "ruff", "format", "--check", *targets],
        ["-m", "mypy", *targets],
        ["-m", "bandit", "-r", "src", "scripts", *targets[3:], "-ll"],
        ["-m", "pytest", "-q"],
        ["-m", "pip", "check"],
    ]
    if args.with_results:
        commands.append(["run_pipeline.py", "validate"])
    checks = []
    for arguments in commands:
        command = [sys.executable, *arguments]
        print("Running: " + " ".join(arguments), flush=True)
        started = time.monotonic()
        result = subprocess.run(command, cwd=root, capture_output=True, text=True, check=False)
        checks.append(
            dict(
                command=command,
                exit_code=result.returncode,
                seconds=round(time.monotonic() - started, 3),
                stdout=result.stdout,
                stderr=result.stderr,
            )
        )
        print(result.stdout, end="", flush=True)
        if result.returncode:
            print(result.stderr, file=sys.stderr, end="", flush=True)
    passed = all(check["exit_code"] == 0 for check in checks)
    write_json(
        root / "outputs/quality/quality_checks.json",
        dict(
            status="passed" if passed else "failed",
            python=sys.version,
            checks=checks,
            scope="local_checks_not_CI_or_absolute_geometry_accuracy",
        ),
    )
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
