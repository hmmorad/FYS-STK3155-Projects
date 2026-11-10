# FYS-STK3155: Machine Learning Projects

Course projects on regression, optimization and neural networks. Both projects use a shared Python environment managed with [uv](https://docs.astral.sh/uv/).

## Projects

### [Project 1: Polynomial regression](project1/README.md)

Polynomial regression on Runge's function using OLS, Ridge and Lasso. We compare closed-form solutions with gradient-based optimization and study model complexity, regularization and generalization.

**Keywords:** OLS, Ridge, Lasso, gradient descent, momentum, AdaGrad, RMSProp, Adam, SGD, bootstrap, bias–variance trade-off, cross-validation.

[Results notebook](project1/notebooks/Project1_results.ipynb) · [Report](project1/report/main.pdf)

### [Project 2: Neural networks from scratch](project2/README.md)

A feed-forward neural network implemented ourselves with NumPy, including dense layers, activation functions, loss functions and backpropagation. We study Runge regression and MNIST classification, including logistic regression as a network without hidden layers.

**Keywords:** feed-forward neural networks, backpropagation from scratch, learned features, activation functions, minibatch training, Adam, RMSProp, L1/L2 regularization, softmax, MNIST, model validation.

[Results notebook](project2/notebooks/Project2_results.ipynb) · [Tests notebook](project2/notebooks/Project2_tests.ipynb) · [Report](project2/report/main.pdf)

## Repository layout

```text
.
├── README.md
├── pyproject.toml           # Shared Python configuration and dependencies
├── uv.lock                 # Locked dependency versions
├── project1/
│   ├── README.md
│   ├── src/project1/       # Regression models, optimizers and resampling
│   ├── notebooks/          # Experiments and plots
│   ├── figures/            # Generated PDF figures
│   └── report/             # LaTeX source, report PDF and generated tables
└── project2/
    ├── README.md
    ├── src/project2/       # Neural network, training and experiments
    ├── tests/              # Gradient and cross-validation checks
    ├── notebooks/          # Results and test notebooks
    ├── datasets/           # MNIST data
    ├── figures/            # Generated PDF figures
    └── report/             # LaTeX source, report PDF and generated tables
```

Project 2 reuses the regression models and shared helpers from Project 1.

## Setup

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/).
2. From the repository root, where `pyproject.toml` and `uv.lock` are located, run:

   ```bash
   uv sync
   ```

3. Open a notebook in an editor with Jupyter support and select the repository's `.venv` as the Python kernel.
4. Run the notebook cells in order. The results notebooks call the experiment functions and save figures and LaTeX tables in their project folders.

The Python requirement and dependencies are defined in `pyproject.toml`. The shared environment is created at the repository root; there is no separate environment for each project.

## Libraries

| Library | Used for |
| --- | --- |
| [NumPy](https://numpy.org/) | Array operations, linear algebra and model implementations |
| [JAX](https://docs.jax.dev/) | Automatic differentiation and gradient checks |
| [Scikit-learn](https://scikit-learn.org/) | Data splitting, resampling, reference models and evaluation metrics |
| [pandas](https://pandas.pydata.org/) | Results tables and LaTeX export |
| [Matplotlib](https://matplotlib.org/) | Figures and heatmaps |
| [Jinja2](https://jinja.palletsprojects.com/) | Template support for pandas LaTeX export |
| [pytest](https://docs.pytest.org/) | Project 2 verification tests |
| [IPython kernel](https://ipykernel.readthedocs.io/) | Running the notebooks in a Jupyter-compatible editor |

The neural network and backpropagation in Project 2 are implemented with NumPy. JAX is used to check gradients, and Scikit-learn provides comparison models.

## Tests

Run the Project 2 tests from the repository root:

```bash
uv run pytest project2/tests
```

The checks can also be run through [Project2_tests.ipynb](project2/notebooks/Project2_tests.ipynb).

## Use of AI tools

GitHub Copilot and ChatGPT were used for code assistance and documentation. Their use is described in the source files and reports. The READMEs were drafted or revised with ChatGPT in November 2026; the original Project 1 README was drafted with GitHub Copilot in October 2026.