import argparse
import logging
import re
from pathlib import Path

import pandas as pd


def _filter_runs(input_file: Path, output_file: Path) -> None:
    """Filter runs to keep only non-aide agents."""
    df = pd.read_json(input_file, lines=True, orient="records", convert_dates=False)
    result_df = df[~df["alias"].str.contains("aide", flags=re.IGNORECASE, regex=True)]
    removed_count = len(df) - len(result_df)
    logging.info(f"Removed {removed_count} runs from {input_file}")
    result_df.to_json(output_file, lines=True, orient="records")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-all-runs", type=Path, required=True)
    parser.add_argument("--output-runs-with-allowed-agents", type=Path, required=True)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level)

    args.output_runs_with_allowed_agents.parent.mkdir(exist_ok=True, parents=True)

    _filter_runs(args.input_all_runs, args.output_runs_with_allowed_agents)


if __name__ == "__main__":
    main()
