# FYS-STK3155 Project 1: Polynomial Regression

Polynomial regression on Runge's function using OLS, Ridge and Lasso. We compare closed-form solutions with gradient-based optimizers and study how polynomial degree, regularization and sample size affect the results.

[Results notebook](notebooks/Project1_results.ipynb) · [Report PDF](report/main.pdf)

## Topics

- OLS, Ridge and Lasso regression.
- Analytical gradients and automatic differentiation with JAX.
- Gradient descent, momentum, AdaGrad, RMSProp, Adam and SGD.
- Bootstrap estimates of prediction error, squared bias and variance.
- K-fold cross-validation and one-standard-error model selection.
- Learning-rate sensitivity, convergence and comparison with closed-form solutions.

## Project layout

```text
project1/
├── README.md
├── src/project1/
│   ├── model_fitter.py     # Combines a model with an optional optimizer
│   ├── models/             # OLS, Ridge and Lasso
│   ├── optimizers/         # Gradient-based update rules
│   ├── resampling/         # Bootstrap and cross-validation
│   ├── scripts/            # Experiment functions: part_a.py to part_i.py
│   ├── utils/              # Data generation, scaling, metrics and export
│   ├── typing.py
│   └── py.typed
├── notebooks/
│   └── Project1_results.ipynb
├── figures/                # Generated PDF figures
└── report/
    ├── main.tex
    ├── main.pdf
    ├── biblio.bib
    └── tables/             # Generated LaTeX tables
```

## Main classes

| Classes | Purpose |
| --- | --- |
| [`BaseModel`, `OLS`, `Ridge`, `Lasso`](src/project1/models/) | Model costs, predictions and gradients. OLS and Ridge provide closed-form solutions; Lasso is fitted iteratively. |
| [`Optimizer` and subclasses](src/project1/optimizers/) | Parameter updates, optimization histories and stopping criteria. |
| [`ModelFitter`](src/project1/model_fitter.py) | Fits a model using either its closed-form solution or a supplied optimizer. |
| [`Bootstrap`](src/project1/resampling/bootstrap.py) | Resampling with replacement and bias–variance estimates. |
| [`CrossValidation`](src/project1/resampling/cross_validation.py) | Validation MSE across folds, its standard error, and degree/regularization sweeps. |

The resampling classes use `ModelFitter`, so the same experiments can use closed-form or gradient-based fitting.

## Running the experiments

Follow the [shared setup instructions](../README.md#setup), running `uv sync` from the repository root. Then open [Project1_results.ipynb](notebooks/Project1_results.ipynb), select the root `.venv` kernel and run the cells in order.

The notebook calls the functions in `src/project1/scripts/part_a.py` through `part_i.py`, produces the plots, and exports figures to `figures/` and tables to `report/tables/`. Experiment settings, including the random seed and parameter grids, are defined in the notebook.

Features are scaled using training-data statistics, and targets are centered before fitting. During cross-validation, these statistics are calculated within each training fold and applied to its validation fold.

Dependencies, Python configuration and the lockfile are shared at the repository root. See the [library overview](../README.md#libraries).

## Use of AI tools

The original README was drafted with GitHub Copilot in October 2026 and revised with ChatGPT in November 2026. Assistance with the code and report is documented in the source files and report.