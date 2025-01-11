import argparse

import dvc.api
import pandas as pd


def _is_task_family_included(task_id: str, task_families: list[str]) -> bool:
    task_family, _ = task_id.split("/", 1)
    return task_family in task_families


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter out runs that are not in the specified task families"
    )
    parser.add_argument("--input-file", help="Input JSONL file")
    parser.add_argument("--output-file", help="Output JSONL file")
    args = parser.parse_args()

    params = dvc.api.params_show(stages="filter_aird_runs")
    task_families = params["stages"]["filter_aird_runs"]["task_families"]

    df = pd.read_json(args.input_file, lines=True, convert_dates=False)
    filtered = df[
        df["task_id"].apply(lambda x: _is_task_family_included(x, task_families))
    ]
    filtered.to_json(args.output_file, orient="records", lines=True)


if __name__ == "__main__":
    main()
