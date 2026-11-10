# FYS-STK3155: Project 1

Polynomial regression (OLS, Ridge, Lasso) on Runge's function, with gradient-based
optimizers, bootstrap resampling and cross-validation. A  See [report/main.tex](report/main.tex) for the report.

## Repository layout

```txt
project1
├── src/project1          
│   ├── models             # OLS, Ridge and Lasso implementations
│   ├── optimizers         # Gradient descent, momentum, AdaGrad, RMSProp, Adam, SGD
│   ├── resampling         # Bootstrap and k-fold cross-validation
│   ├── scripts            # part_a.py … part_i.py, one per project subtask
│   ├── utils              # Metrics, scaling, plotting helpers, LaTeX export
│   └── py.typed           # Marker file telling type checkers the package is typed
├── notebooks
│   └── Project1_results.ipynb   # Runs the scripts and test, and produces all figures/tables
├── figures                # Generated plots (.pdf), written by the notebook
├── report
│   ├── main.tex           # Report source
│   ├── biblio.bib
│   └── tables              # Generated LaTeX tables, written by the notebook
├── tests
├── main.py
├── pyproject.toml         # Project metadata, dependencies and build configuration
└── uv.lock                # Locked dependency versions, produced by uv
```

## Classes

We took a class-oriented approach for the regression models, optimizers and resampling, so here's a quick overview of the main classes (docstrings in the source have more detail).

#### `BaseModel`, `OLS`, `Ridge`, `Lasso` ([models/](src/project1/models/))

`BaseModel` holds the design matrix `X`, targets `y`, parameters `theta`, and the regularization strength `lam`. It implements the cost function, analytical gradient, a JAX-based gradient (via `jax.grad`), and the Hessian eigenvalues, shared by all three models. `OLS`, `Ridge` and `Lasso` are subclasses, and each add a `closed_form()` method (`Lasso.closed_form()` and `Lasso.hessian_eigs()`raises `NotImplementedError`).

To fit a model directly:

```python
model = Ridge(X, y, lam=0.1)
theta = model.closed_form()
```

#### `Optimizer` and its subclasses ([optimizers/](src/project1/optimizers/))

`Optimizer` is an abstract base class defining `step()` (one parameter update) and `reset()`, which reset the state. `GradientDescent`, `Momentum`, `AdaGrad`, `RMSProp`, `Adam` and `SGD` each implement their own update rule. The base class also provides `optimize(model, theta0, max_iter, tol, ...)`, which repeatedly calls `step()` using the model's gradient until convergence or `max_iter` is reached, and returns an `OptimizationResult` with the parameter history and cost per iteration.

#### `ModelFitter` ([model_fitter.py](src/project1/model_fitter.py))

A thin wrapper that ties a model class (e.g. `OLS`) together with an optional `Optimizer`. Calling `fit(X, y)` either computes the closed-form solution or runs the optimizer, depending on whether an optimizer was supplied. This lets the same resampling and cross-validation code run with closed-form or gradient-based fitting.

```python
fitter = ModelFitter(Ridge, model_kwargs={"lam": 0.1}, optimizer=Adam())
fitter.fit(X_train, y_train)
y_pred = fitter.predict(X_test)
```

#### `Bootstrap` and `CrossValidation` ([resampling/](src/project1/resampling/))

Both classes take a `ModelFitter` instance (instead of a model directly) so that they work regardless of the underlying fitting method. `Bootstrap.run(degree)` resamples the training data with replacement and returns a `BootstrapResult` (MSE, bias², variance); `bias_variance_sweep(max_degree)` repeats this per polynomial degree. `CrossValidation` similarly splits the data into `k` folds (via scikit-learn's `KFold`) and reports the mean MSE, $R^2$ and standard error across folds.

## Running the project

This project uses [UV](https://docs.astral.sh/uv/) to manage the Python version and dependencies, and uv is required to run it. UV is fast becoming the standard tool for Python packaging/dependency management, replacing the pip + venv + requirements.txt workflow.

1. Install uv (see the [installation guide](https://docs.astral.sh/uv/getting-started/installation/)).
2. From the project root, run:

   ```bash
   uv sync
   ```

   This creates a `.venv` virtual environment and installs the exact dependency
   versions from `uv.lock`.
3. Open [notebooks/Project1_results.ipynb](notebooks/Project1_results.ipynb) and select
   the `.venv` kernel created by `uv sync`. Running the notebook reproduces all the
   figures in `figures/` and the tables in `report/tables/`.


## Libraries used

- [NumPy](https://numpy.org/): array operations and linear algebra
- [JAX](https://jax.readthedocs.io/): automatic differentiation
- [scikit-learn](https://scikit-learn.org/): `KFold()`, `Resample()` and `train_test_split()`, and `Lasso` to compare
- [pandas](https://pandas.pydata.org/): tabular results and LaTeX table export
- [matplotlib](https://matplotlib.org/): figures
- [Jinja2](https://jinja.palletsprojects.com/): LaTeX table templating

## Use of AI tools

This README was generated with GitHub Copilot (October, 2026).
- Role: generated the full content of the README (repository layout, classes overview, running instructions and libraries used)
- Modifications: I modified many of the comments, and how the classes are built. It gave a good skeleton.
- Prompt: Make a repository layout, classes overview, running instructions and libraries used (use `tree` in the terminal for project structure). Say that this project uses UV.

For the rest of the project (code and report), we used GitHub Copilot autocomplete and ChatGPT.

