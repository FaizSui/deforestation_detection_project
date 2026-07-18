"""
Step 13: Assemble the per-section markdown files into one manuscript.

The individual files in paper/ are the source of truth; this script
concatenates them in journal order into paper/manuscript.md and, if
pandoc is available (directly or via the pypandoc_binary package),
renders paper/manuscript.docx with the LaTeX math converted to Word
equations. Figures are embedded, each with its caption, in a Figures
section at the end -- the layout most student journals expect.

Re-run this whenever any section file changes.
"""

import os
import re

PAPER_DIR = os.path.join(os.path.dirname(__file__), "..", "paper")
FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figures", "publication")

# Order in which sections appear in the assembled manuscript.
SECTION_ORDER = [
    "title_page.md",
    "abstract.md",
    "introduction.md",
    "methods.md",
    "results.md",
    "discussion.md",
    "conclusion.md",
    "acknowledgments.md",
    "references.md",
    "reproducibility.md",
]

FIGURES = [
    ("fig1_confusion_matrix.png", 1),
    ("fig2_landcover_maps.png", 2),
    ("fig3_change_gfw_overlay.png", 3),
    ("fig4_threshold_sensitivity.png", 4),
]


def strip_wordcount_annotations(text):
    # Remove editorial "(180 words)" markers and HTML comment notes.
    text = re.sub(r"\n\*\(\d+ words\)\*\n?", "\n", text)
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    return text.strip()


def load_captions():
    """Return {fig_number: caption_markdown} parsed from captions.md."""
    path = os.path.join(PAPER_DIR, "captions.md")
    with open(path, encoding="utf-8") as f:
        text = f.read()
    captions = {}
    for m in re.finditer(r"\*\*Figure (\d+)\.(.*?)(?=\n\n\*\*Figure |\Z)", text, flags=re.DOTALL):
        num = int(m.group(1))
        captions[num] = ("**Figure " + m.group(1) + "." + m.group(2)).strip()
    return captions


def assemble():
    parts = []
    for fname in SECTION_ORDER:
        path = os.path.join(PAPER_DIR, fname)
        if not os.path.exists(path):
            print(f"  WARNING: {fname} missing, skipping")
            continue
        with open(path, encoding="utf-8") as f:
            parts.append(strip_wordcount_annotations(f.read()))

    captions = load_captions()
    fig_section = ["# Figures"]
    for fname, num in FIGURES:
        rel = os.path.join("..", "figures", "publication", fname).replace("\\", "/")
        fig_section.append(f"![]({rel})\n")
        fig_section.append(captions.get(num, f"**Figure {num}.**"))
    parts.append("\n\n".join(fig_section))

    manuscript = "\n\n".join(parts) + "\n"
    out_md = os.path.join(PAPER_DIR, "manuscript.md")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(manuscript)
    words = len(re.sub(r"[#*|`\-]", " ", manuscript).split())
    print(f"Wrote {out_md} (~{words} words including tables/captions)")
    return out_md


def to_docx(md_path):
    out_docx = os.path.join(PAPER_DIR, "manuscript.docx")
    try:
        import pypandoc
    except ImportError:
        print("pypandoc not available; wrote manuscript.md only.")
        print("To produce the .docx: pip install pypandoc_binary, then re-run.")
        return
    try:
        # pandoc converts LaTeX math to native Word equations (OMML)
        # automatically for docx output; no math flag needed.
        pypandoc.convert_file(md_path, "docx", outputfile=out_docx,
                              extra_args=[f"--resource-path={PAPER_DIR}"])
        print(f"Wrote {out_docx}")
    except Exception as e:
        print(f"pandoc conversion failed: {e}")
        print("manuscript.md is still valid; convert manually once pandoc is installed.")


if __name__ == "__main__":
    md = assemble()
    to_docx(md)
