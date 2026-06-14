"""Command line entry point for GeoGram."""

from pathlib import Path
import argparse

from .data import load_municipalities, save_filtered_ice


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Load Czech municipalities and filter names ending in -ice."
    )
    parser.add_argument(
        "input_path",
        help="Path to the raw municipality CSV file.",
    )
    parser.add_argument(
        "--output",
        default="data/processed/municipalities_ice.csv",
        help="Path to save the filtered output CSV file.",
    )
    parser.add_argument(
        "--name-column",
        default="name",
        help="Column that contains municipality names.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    input_path = Path(args.input_path)
    output_path = Path(args.output)

    df = load_municipalities(input_path)
    filtered = save_filtered_ice(df, output_path, name_column=args.name_column)
    print(f"Loaded {len(df):,} rows and saved {len(filtered):,} rows to {output_path}")


if __name__ == "__main__":
    main()
