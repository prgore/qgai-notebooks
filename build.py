"""Build the .ipynb notebooks from src/*.py (cells split by '# %%' markers) and execute them.

Usage:  python build.py            # build + execute all
        python build.py 03         # only notebooks whose name starts with 03
"""
import sys, time, pathlib, nbformat
from nbclient import NotebookClient

ROOT = pathlib.Path(__file__).parent
SRC, OUT = ROOT / "src", ROOT / "notebooks"

def parse(path):
    cells, kind, buf = [], None, []
    def flush():
        text = "\n".join(buf).strip("\n")
        if kind and text:
            if kind == "md":
                text = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in text.splitlines())
                cells.append(nbformat.v4.new_markdown_cell(text))
            else:
                cells.append(nbformat.v4.new_code_cell(text))
    for line in path.read_text().splitlines():
        if line.startswith("# %%"):
            flush(); buf = []
            kind = "md" if "[markdown]" in line else "code"
        else:
            buf.append(line)
    flush()
    return cells

def main(prefix=""):
    for src in sorted(SRC.glob(f"{prefix}*.py")):
        nb = nbformat.v4.new_notebook()
        nb.cells = parse(src)
        nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
        nb.metadata["language_info"] = {"name": "python"}
        t = time.time()
        NotebookClient(nb, timeout=900, kernel_name="python3", resources={"metadata": {"path": str(OUT)}}).execute()
        out = OUT / (src.stem + ".ipynb")
        nbformat.write(nb, out)
        print(f"{out.name}: {len(nb.cells)} cells, executed in {time.time() - t:.0f}s")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "")
