import shutil
from pathlib import Path

from dvc.repo import Repo


def _get_changed_plots(old_branch: str, new_branch: str) -> list[str]:
    repo = Repo()
    diff_dict = repo.diff(old_branch, new_branch)

    changed_plots_paths = []
    for modified_dict in diff_dict.get("modified", []):
        modified_path = modified_dict.get("path", "")
        if modified_path.startswith("plots/"):
            changed_plots_paths.append(modified_path)

    return changed_plots_paths


def _setup_image_ref(
    branch_name: str, plot_path: str, plots_dir: Path, output_dir: Path
) -> str:
    branch_plot = (
        plots_dir / f"{branch_name.replace('/', '_')}_{plot_path.replace('/', '_')}"
    )
    shutil.copy2(branch_plot, output_dir / branch_plot.name)
    return f"![](./{branch_plot.name})"


def main(plots_dir: str, output_dir: str, old_branch: str, new_branch: str) -> None:
    plots_path = Path(plots_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    changed_plots = _get_changed_plots(old_branch, new_branch)

    summary_file = output_path / "summary.md"
    with summary_file.open("w") as f:
        f.write("# Plot Comparison Report\n\n")

        print("Changed plots:")
        for plot_path in changed_plots:
            print(plot_path)

            # Write table header with plot name
            f.write(f"### {plot_path}\n\n")
            f.write(" old | new\n")
            f.write(":-:|:-:\n")  # Center-align both columns
            f.write(
                f"{_setup_image_ref(old_branch, plot_path, plots_path, output_path)} | {_setup_image_ref(new_branch, plot_path, plots_path, output_path)}\n\n"
            )

        if not changed_plots:
            f.write("No changed plots found \n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compare plot files")
    parser.add_argument(
        "--plots-dir",
        default="dvc_plots/static",
        type=Path,
        help="Directory containing plot files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default="changed_plots",
        help="Output directory for changed plots summary",
    )
    parser.add_argument(
        "--old-branch", type=str, required=True, help="First branch name"
    )
    parser.add_argument(
        "--new-branch", type=str, required=True, help="Second branch name"
    )

    args = parser.parse_args()
    main(args.plots_dir, args.output_dir, args.old_branch, args.new_branch)
