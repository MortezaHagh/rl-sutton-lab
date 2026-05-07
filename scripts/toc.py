# Auto-generated Table of Contents from notebook markdown headers
# Re-run this cell after adding/changing headings.

import json
import re
from pathlib import Path
from IPython.display import Markdown, display


def generate_toc(path: Path) -> Markdown:
    nb_path = Path(path)
    if not nb_path.exists():
        display(Markdown("**TOC error:** could not find `ch_2_Bandits.ipynb` in current folder."))
    else:
        nb = json.loads(nb_path.read_text(encoding="utf-8"))

        heading_re = re.compile(r"^(#{1,6})\s+(.+?)\s*$")

        def slugify(text: str) -> str:
            # Match common markdown heading anchor behavior.
            text = text.strip().lower()
            text = re.sub(r"[`*_~]", "", text)
            text = re.sub(r"[^a-z0-9\-\s]", "", text)
            text = re.sub(r"\s+", "-", text)
            text = re.sub(r"-+", "-", text).strip("-")
            return text

        items = []
        for cell in nb.get("cells", []):
            if cell.get("cell_type") != "markdown":
                continue
            source = "".join(cell.get("source", []))
            for line in source.splitlines():
                m = heading_re.match(line)
                if m:
                    level = len(m.group(1))
                    title = m.group(2).strip()
                    items.append((level, title))

        if not items:
            display(Markdown("## Table of Contents\n\n_No markdown headers found yet._"))
        else:
            toc_lines = ["## Table of Contents", ""]
            base_level = min(level for level, _ in items)

            for level, title in items:
                indent = "  " * (level - base_level)
                anchor = slugify(title)
                toc_lines.append(f"{indent}- [{title}](#{anchor})")

            display(Markdown("\n".join(toc_lines)))
