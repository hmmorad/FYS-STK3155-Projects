"""Save wide LaTeX tables for Project 2 using Project 1 formatting.

Tools used:
    - Copilot autocomplete
"""

from pathlib import Path

import pandas as pd  # type: ignore
from project1.utils.report import save_latex_table


def save_wide_table(
    df: pd.DataFrame, tables_dir: Path, filename: str, caption: str, label: str
) -> Path | None:
    """Use Project 1 formatting and span both LaTeX columns."""
    path = save_latex_table(df, tables_dir, filename, caption, label)
    if path is not None:
        latex = path.read_text(encoding="utf-8")
        latex = latex.replace(r"\begin{table}[htbp]", r"\begin{table*}[t]")
        latex = latex.replace(r"\end{table}", r"\end{table*}")
        path.write_text(latex, encoding="utf-8")
    return path
