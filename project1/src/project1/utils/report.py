"""Utility functions for saving figures and LaTeX tables.

Tools used:
- Copilot autocomplete
"""

from pathlib import Path

import matplotlib.figure
import pandas as pd  # type: ignore


def save_figure(
    fig: matplotlib.figure.Figure, figures_dir: Path | None, filename: str
) -> Path | None:
    """Save a figure as a PDF."""
    if figures_dir is None:
        return None
    path = figures_dir / filename
    fig.dpi = 300
    fig.savefig(path, bbox_inches="tight")

    print(f"Saved: {path}")
    return path


def save_latex_table(
    df: pd.DataFrame, tables_dir: Path | None, filename: str, caption: str, label: str
) -> Path | None:
    """Save a DataFrame as a LaTeX table.

    LLM-assisted
    ------------
    Tool: Github Copilot (September 2026)
    Role: generated the whole function on my behalf
    Modifications: modified the LaTeX table to use hline instead of top/mid/bottomrule

    """
    if tables_dir is None:
        return None
    path = tables_dir / filename
    latex = df.to_latex(index=False, escape=False, float_format=lambda value: f"{value:.4g}")
    latex = (
        latex.replace("\\toprule", "\\hline\\hline")
        .replace("\\midrule", "\\hline")
        .replace("\\bottomrule", "\\hline\\hline")
    )
    text = (
        "\\begin{table}[htbp]\n"
        "    \\centering\n"
        f"    \\caption{{{caption}}}\n"
        f"    \\label{{tab:{label}}}\n"
        f"{latex}"
        "\\end{table}\n"
    )

    path.write_text(text, encoding="utf-8")

    print(f"Saved: {path}")
    print(f"Use in main.tex: \\input{{tables/{path.name}}}")

    return path
