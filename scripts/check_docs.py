"""Check documentation sources and execute standalone Quarto Python examples."""
from __future__ import annotations

import inspect
import os
import re
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit

os.environ.setdefault("MPLBACKEND", "Agg")

import nbformat
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    os.chdir(ROOT)
    config = yaml.safe_load((ROOT / "_quarto.yml").read_text())
    for source in config["project"]["render"]:
        if not (ROOT / source).is_file():
            raise ValueError(f"Quarto render source missing: {source}")
    qmd_sources = sorted(set(ROOT.glob("*.qmd")) | {
        ROOT / source for source in config["project"]["render"]
        if Path(source).suffix == ".qmd"
    })
    texts = {}
    for path in [ROOT / "README.md", *qmd_sources, *ROOT.glob("data/*.md")]:
        texts[path] = path.read_text()
    # Notebook editors also create .ipynb_checkpoints and local old/ copies.
    # Validate published sources, not those unrendered editing backups.
    notebooks = sorted(ROOT / source for source in config["project"]["render"]
                       if Path(source).suffix == ".ipynb")
    for path in notebooks:
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        code = [c for c in notebook.cells if c.cell_type == "code"]
        if not code or any(c.execution_count is None for c in code):
            raise ValueError(f"Notebook has unexecuted cells: {path.relative_to(ROOT)}")
        if any(o.output_type == "error" for c in code for o in c.outputs):
            raise ValueError(f"Notebook has execution errors: {path.relative_to(ROOT)}")
        texts[path] = "\n".join(c.source for c in notebook.cells if c.cell_type == "markdown")
    bibliography = (ROOT / config["bibliography"]).read_text()
    keys = set(re.findall(r"@\w+\{([^,]+),", bibliography))
    problems = []
    links = 0
    for path, source in texts.items():
        # Ignore examples inside code fences when checking prose links/citations.
        prose = re.sub(r"```.*?```", "", source, flags=re.S)
        for target in re.findall(r"\]\(([^)\s]+)(?:\s+[^)]*)?\)", prose):
            parsed = urlsplit(target.strip("<>"))
            if parsed.scheme or parsed.netloc or not parsed.path:
                continue
            destination = (path.parent / unquote(parsed.path)).resolve()
            links += 1
            if not destination.is_file():
                problems.append(f"{path.relative_to(ROOT)}: missing link {target}")
        for key in re.findall(r"(?<![\w.])@([\w-]+)", prose):
            if key not in keys:
                problems.append(f"{path.relative_to(ROOT)}: unknown citation {key}")
    from PRESS import PRESS
    api = (ROOT / "api.qmd").read_text()
    for name, method in inspect.getmembers(PRESS, predicate=callable):
        if not name.startswith("_") and f"{name}(" not in api:
            problems.append(f"api.qmd: undocumented method {name}")
    if problems:
        raise SystemExit("\n".join(problems))
    example_count = 0
    for path in qmd_sources:
        blocks = re.findall(r"```\{python[^}]*\}\n(.*?)\n```", path.read_text(), flags=re.S)
        namespace = {"__name__": "__main__"}
        for block in blocks:
            exec(compile(block, str(path), "exec"), namespace)
            example_count += 1
    print(f"Checked {len(notebooks)} executed notebooks, {links} local links, public methods and citations; executed {example_count} Quarto examples.")


if __name__ == "__main__":
    main()
