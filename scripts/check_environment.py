"""Verify installed package versions against the exact project pins."""
from __future__ import annotations

import argparse
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import tomllib


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extras", default="", help="Comma-separated installed extras to verify")
    args = parser.parse_args()
    config = tomllib.loads((Path(__file__).resolve().parents[1] / "pyproject.toml").read_text())
    dependencies = list(config["project"]["dependencies"])
    for extra in filter(None, args.extras.split(",")):
        if extra not in config["project"]["optional-dependencies"]:
            parser.error(f"Unknown extra: {extra}")
        dependencies.extend(config["project"]["optional-dependencies"][extra])
    errors = []
    for dependency in dependencies:
        name, expected = dependency.split("==")
        try:
            actual = version(name)
        except PackageNotFoundError:
            actual = "not installed"
        if actual != expected:
            errors.append(f"{name}: expected {expected}, found {actual}")
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"Verified {len(dependencies)} exact dependency pins.")


if __name__ == "__main__":
    main()
