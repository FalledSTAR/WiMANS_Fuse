import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from utils import append_csv_rows, compact_prediction_rows  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Create a compact prediction-vs-label CSV.")
    parser.add_argument("--input", required=True, help="Detailed prediction CSV, usually val_predictions_best.csv.")
    parser.add_argument("--output", default=None, help="Output CSV. Defaults to *_compact.csv beside input.")
    return parser.parse_args()


def read_csv_rows(path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def default_output_path(input_path):
    return input_path.with_name(f"{input_path.stem}_compact{input_path.suffix}")


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else default_output_path(input_path)
    rows = read_csv_rows(input_path)
    compact_rows = compact_prediction_rows(rows)
    output_path.unlink(missing_ok=True)
    if compact_rows:
        append_csv_rows(output_path, compact_rows, list(compact_rows[0].keys()))
    print(f"saved_compact_predictions={output_path}")


if __name__ == "__main__":
    main()
