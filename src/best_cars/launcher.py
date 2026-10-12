"""Launch the local Streamlit explorer with named settings."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

from best_cars.storage import data_directory


def main(*, argv: list[str] | None = None) -> int:
    """Launch Streamlit on localhost by default.

    Args:
        argv: Optional named CLI settings for tests.

    Returns:
        Streamlit process exit code.
    """
    parser = argparse.ArgumentParser(
        description="Launch the Best Cars Streamlit explorer"
    )
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--server-port", type=int, default=8501)
    parser.add_argument("--server-address", default="127.0.0.1")
    args = parser.parse_args(args=argv)
    if not 1 <= args.server_port <= 65535:
        parser.error("--server-port must be between 1 and 65535")
    environment = os.environ.copy()
    environment["BEST_CARS_DATA_DIR"] = str(data_directory(supplied=args.data_dir))
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(Path(__file__).with_name("app.py")),
        "--server.address",
        args.server_address,
        "--server.port",
        str(args.server_port),
        "--browser.gatherUsageStats",
        "false",
    ]
    return subprocess.call(args=command, env=environment)


if __name__ == "__main__":
    raise SystemExit(main())
