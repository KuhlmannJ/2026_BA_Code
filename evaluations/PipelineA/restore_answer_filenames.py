"""Restore the original answer filenames in the SysP answer folders.

The SysP runs wrote their JSON files with a sanitized stem (every character
outside [A-Za-z0-9_-] replaced by '_'), so names like

    "Fresenius SE_2019_report.json"  ->  "Fresenius_SE_2019_report.json"
    "bekaert (d) sa_2022_report.json" -> "bekaert__d__sa_2022_report.json"
    "h. lundbeck_2021_report.json"   ->  "h__lundbeck_2021_report.json"

no longer match the reference folder. This script rebuilds the mapping by
applying the same sanitization to the reference names and renaming back.
"""

import argparse
import re
from pathlib import Path

BASE = Path(__file__).parent
REFERENCE = BASE / "PipelineA-Answers" / "bare"
TARGETS = [
    BASE / "PipelineA-Answers" / "thinking",
    BASE / "PipelineA-Answers" / "thinking_system",
]


def sanitize(name: str) -> str:
    p = Path(name)
    return re.sub(r"[^A-Za-z0-9_-]", "_", p.stem) + p.suffix


def build_mapping(reference: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for path in reference.iterdir():
        if path.name.startswith("."):
            continue
        key = sanitize(path.name)
        if key in mapping:
            raise ValueError(
                f"Ambiguous mapping: '{mapping[key]}' and '{path.name}' both sanitize to '{key}'"
            )
        mapping[key] = path.name
    return mapping


def restore(folder: Path, mapping: dict[str, str], dry_run: bool) -> None:
    print(f"\n=== {folder.name} ===")
    renamed = unchanged = skipped = 0
    for path in sorted(folder.iterdir()):
        if path.name.startswith("."):
            continue
        target_name = mapping.get(path.name)
        if target_name is None:
            print(f"  [skip]  no reference name for '{path.name}'")
            skipped += 1
            continue
        if target_name == path.name:
            unchanged += 1
            continue
        target = folder / target_name
        if target.exists():
            print(f"  [skip]  target already exists: '{target_name}'")
            skipped += 1
            continue
        print(f"  {path.name}  ->  {target_name}")
        if not dry_run:
            path.rename(target)
        renamed += 1
    print(f"  renamed: {renamed}, already correct: {unchanged}, skipped: {skipped}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="actually rename the files (default is a dry run)",
    )
    args = parser.parse_args()

    mapping = build_mapping(REFERENCE)
    print(f"Reference names: {len(mapping)} ({REFERENCE.name})")
    if not args.apply:
        print("DRY RUN - pass --apply to rename")

    for folder in TARGETS:
        if not folder.is_dir():
            print(f"\n=== {folder.name} === missing, skipped")
            continue
        restore(folder, mapping, dry_run=not args.apply)


if __name__ == "__main__":
    main()
