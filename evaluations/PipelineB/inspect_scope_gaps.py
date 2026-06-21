from __future__ import annotations

import argparse
import json
from pathlib import Path


EXPECTED_SCOPES = [
    "scope_1",
    "scope_2_location_based",
    "scope_2_market_based",
    "scope_3",
]


def format_path(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd()))
    except Exception:
        return str(path)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def print_emissions_block(path: Path) -> None:
    try:
        data = load_json(path)
    except Exception as exc:
        print(f"  [ERR] could not load JSON: {exc}")
        return

    emissions = data.get("emissions", None)
    print(f"  emissions type: {type(emissions)}")
    if emissions is None:
        print("  emissions: None")
        return

    print(json.dumps(emissions, indent=2, ensure_ascii=False))


def inspect_folder(folder: Path, sample_emissions: int) -> None:
    files = sorted(folder.rglob("*.json"))
    print("\n" + "=" * 80)
    print(f"Folder: {format_path(folder)} — {len(files)} JSON files")

    file_scopes: dict[Path, set[str]] = {}
    for path in files:
        try:
            data = load_json(path)
        except Exception as exc:
            print(f"  [ERR] could not read {format_path(path)}: {exc}")
            file_scopes[path] = set()
            continue

        emissions = data.get("emissions", {})
        if isinstance(emissions, dict):
            file_scopes[path] = set(emissions.keys())
        else:
            file_scopes[path] = set()

    missing_by_scope: dict[str, list[Path]] = {}
    for scope in EXPECTED_SCOPES:
        missing = [path for path, scopes in file_scopes.items() if scope not in scopes]
        missing_by_scope[scope] = missing
        present = len(files) - len(missing)
        print(f"  {scope}: present in {present}/{len(files)} files; missing in {len(missing)} files")
        for path in missing:
            print(f"     {format_path(path)}")

    if sample_emissions <= 0:
        return

    sample_files = []
    seen = set()
    for scope in EXPECTED_SCOPES:
        for path in missing_by_scope[scope]:
            if path not in seen:
                sample_files.append(path)
                seen.add(path)
            if len(sample_files) >= sample_emissions:
                break
        if len(sample_files) >= sample_emissions:
            break

    if sample_files:
        print("\n  Sample emissions blocks from files with missing scopes:")
        for path in sample_files:
            print("\n" + "-" * 80)
            print(f"  File: {format_path(path)}")
            print_emissions_block(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect missing scope keys and optionally print sample emissions blocks."
    )
    parser.add_argument(
        "--base-dir",
        default="evaluations/PipelineB/PipelineB-Answers",
        help="Base directory containing the pipeline answer folders.",
    )
    parser.add_argument(
        "--sample-emissions",
        type=int,
        default=0,
        help="Print emissions blocks for this many sample files with missing scopes.",
    )
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    if not base_dir.exists():
        raise SystemExit(f"Base folder not found: {base_dir}")

    folders = sorted(path for path in base_dir.iterdir() if path.is_dir())
    for folder in folders:
        inspect_folder(folder, args.sample_emissions)


if __name__ == "__main__":
    main()