"""
Package entry point:

    python -m milk10k_pipeline

Regenerates all Part 1, Part 2 and summary artifacts.
"""
from .metadata_analysis import run_part1, load_metadata
from .color_analysis import run_part2
from .export_summaries import run as run_summaries

METADATA = "data/raw/milk10k/metadata.csv"
IMAGES = "data/raw/milk10k/images"
FIG = "reports/figures_task2"


def main():
    print("[1/3] Part 1: metadata analysis")
    run_part1(METADATA, FIG)

    print("[2/3] Part 2: color analysis")
    df = load_metadata(METADATA)
    run_part2(df, IMAGES, FIG, n_per_class=25, seed=0)

    print("[3/3] Summary tables")
    run_summaries(METADATA, FIG, IMAGES)

    print("Done. Figures and summaries in:", FIG)


if __name__ == "__main__":
    main()
