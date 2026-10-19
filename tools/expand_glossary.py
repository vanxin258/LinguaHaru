import argparse
from pathlib import Path

from core.engine.glossary_expander import expand_glossary_csv


def build_parser():
    parser = argparse.ArgumentParser(description="Expand a glossary CSV with controlled English variants.")
    parser.add_argument("input", help="Input glossary CSV file")
    parser.add_argument(
        "-o",
        "--output",
        help="Output CSV file. Defaults to <input>_expanded.csv",
    )
    parser.add_argument(
        "--src-lang",
        default="en",
        help="Source language column name to expand (default: en)",
    )
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path.with_name(f"{input_path.stem}_expanded{input_path.suffix}")

    count = expand_glossary_csv(str(input_path), str(output_path), src_lang=args.src_lang)
    print(f"Expanded {count} glossary rows to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
