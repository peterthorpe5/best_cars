"""Exercise every app page against a selected full local dataset."""

import argparse
import os
from pathlib import Path

from streamlit.testing.v1 import AppTest

from best_cars.app import PAGES


def main(*, argv: list[str] | None = None) -> int:
    """Check full-data page rendering without external service calls.

    Args:
        argv: Optional named CLI settings.

    Returns:
        Zero when all pages render without exceptions.

    Raises:
        RuntimeError: If a page raises a Streamlit exception.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args(args=argv)
    os.environ["BEST_CARS_DATA_DIR"] = str(args.data_dir.resolve())
    app = AppTest.from_file(
        script_path=Path(__file__).parents[1] / "app.py", default_timeout=60
    ).run()
    for page in PAGES:
        app.radio(key="page").set_value(v=page).run()
        if app.exception:
            raise RuntimeError(
                f"{page}: {[exception.message for exception in app.exception]}"
            )
        print(f"PASS\t{page}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
