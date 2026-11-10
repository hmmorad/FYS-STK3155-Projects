"""Part d: regularization, model selection and regression comparison.

Tools used:
    - Copilot autocomplete
    - ChatGPT (November 2026)


LLM-Assisted:

------------
Tool: ChatGPT (November 2026)
Role: Helped with skeleton code for part d.
Modifications: Fitted our implementation. Typing, and adjusted for our setup.
Promt: Explained what figures we wanted to plot, and asked it to give skeleton code for part d.


"""

from __future__ import annotations

from typing import Any

import numpy as np
from project1.model_fitter import ModelFitter
from project1.models import OLS, Lasso, Ridge
from project1.optimizers.adam import Adam as PolynomialAdam
from project1.utils import fit_polynomial, predict_polynomial, scale_data
from project1.utils.metrics import MSE, R2

from project2.network.activations import Identity, LReLU, ReLU, Sigmoid
from project2.optimizers.adam import Adam
from project2.training.cross_validation import CrossValidation
from project2.training.network_trainer import NetworkTrainer
from project2.typing import Array

ACTIVATIONS = {"Sigmoid": Sigmoid, "ReLU": ReLU, "Leaky ReLU": LReLU}


def run(
    x_train: Array,
    y_train: Array,
    architectures: list[tuple[int, ...]],
    gammas: Array,
    lambdas: Array,
    epochs: int = 500,
    batch_size: int = 16,
    n_splits: int = 5,
    seed: int = 2026,
) -> dict[str, Any]:
    """Search penalties and apply one SE per activation/penalty.

    LLM-Assisted:
    ----
    See top of part_d.py for more details.

    Args:
        x_train: Training input data.
        y_train: Training target data.
        architectures: List of network architectures to evaluate.
        gammas: List of learning rates for the Adam optimizer.
        lambdas: List of regularization penalties to evaluate.
        epochs: Number of training epochs.
        batch_size: Size of each training batch.
        n_splits: Number of cross-validation splits.
        seed: Random seed for reproducibility.

    Returns:
        Dictionary containing cross-validation results and selected model configurations.
    """
    shape = (len(architectures), len(lambdas), len(gammas))
    parameters = []
    for hidden in architectures:
        sizes = (x_train.shape[1], *hidden, 1)
        parameters.append(sum((a + 1) * b for a, b in zip(sizes[:-1], sizes[1:], strict=True)))

    results: dict[str, Any] = {}

    for name, activation in ACTIVATIONS.items():
        results[name] = {}

        for penalty in ("L1", "L2"):
            means = np.full(shape, np.inf)
            errors = np.full(shape, np.inf)

            for a, hidden in enumerate(architectures):
                for lmb_idx, lmbd in enumerate(lambdas):
                    for g, gamma in enumerate(gammas):
                        cv = CrossValidation(
                            network_kwargs={
                                "n_inputs": x_train.shape[1],
                                "layers": (
                                    tuple((nodes, activation) for nodes in hidden)
                                    + ((1, Identity),)
                                ),
                                "lmbd1": (float(lmbd) if penalty == "L1" else 0.0),
                                "lmbd2": (float(lmbd) if penalty == "L2" else 0.0),
                                "seed": seed,
                            },
                            optimizer=Adam,
                            optimizer_kwargs={"gamma": float(gamma)},
                            n_splits=n_splits,
                            seed=seed,
                            scale_features=True,
                        )

                        result = cv.run(
                            x_train,
                            y_train,
                            trainer_kwargs={
                                "epochs": epochs,
                                "batch_size": batch_size,
                                "seed": seed,
                            },
                        )

                        # The network's regression loss is half-MSE.
                        mse = 2 * result.mean_score
                        se = 2 * result.standard_error

                        if np.isfinite(mse) and np.isfinite(se):
                            means[a, lmb_idx, g] = mse
                            errors[a, lmb_idx, g] = se

            valid = np.isfinite(means) & np.isfinite(errors)
            if not valid.any():
                raise ValueError(f"No finite CV score for {name}, {penalty}.")

            minimum = tuple(
                int(i) for i in np.unravel_index(np.argmin(np.where(valid, means, np.inf)), shape)
            )

            threshold = float(means[minimum] + errors[minimum])
            within = valid & (means <= np.nextafter(threshold, np.inf))
            candidates = [tuple(int(i) for i in index) for index in np.argwhere(within)]

            # Prefer fewer parameters, fewer layers, then larger lambda.
            selected = min(
                candidates,
                key=lambda i: (
                    parameters[i[0]],
                    len(architectures[i[0]]),
                    -lambdas[i[1]],
                    means[i],
                ),
            )

            a, lam, g = selected
            config = {
                "activation": name,
                "penalty": penalty,
                "hidden_sizes": tuple(architectures[a]),
                "gamma": float(gammas[g]),
                "lambda": float(lambdas[lam]),
                "n_parameters": parameters[a],
                "cv_mse": float(means[selected]),
                "cv_se": float(errors[selected]),
            }

            results[name][penalty] = {
                "cv_mse": means,
                "cv_se": errors,
                "selection": {
                    "minimum": minimum,
                    "selected": selected,
                    "within": within,
                    "threshold": threshold,
                },
                "selected": config,
            }

    # Compare the three activation-specific one-SE choices.
    best = {
        penalty: min(
            (results[name][penalty]["selected"] for name in ACTIVATIONS),
            key=lambda config: config["cv_mse"],
        )
        for penalty in ("L1", "L2")
    }

    return {"results": results, "best": best}


def fit_selected(
    x_train: Array,
    y_train: Array,
    config: dict[str, Any],
    epochs: int = 500,
    batch_size: int = 16,
    seed: int = 2026,
) -> dict[str, Any]:
    """Refit a selected network on all training observations.

    LLM-Assisted:
    ----
    See top of part_d.py for more details.

    """
    Xs, xm, xs, yc, ym = scale_data(x_train, y_train)
    activation = ACTIVATIONS[config["activation"]]
    lmbd = config["lambda"]

    trainer = NetworkTrainer(
        network_kwargs={
            "n_inputs": Xs.shape[1],
            "layers": (
                tuple((nodes, activation) for nodes in config["hidden_sizes"]) + ((1, Identity),)
            ),
            "lmbd1": lmbd if config["penalty"] == "L1" else 0.0,
            "lmbd2": lmbd if config["penalty"] == "L2" else 0.0,
            "seed": seed,
        },
        optimizer=Adam,
        optimizer_kwargs={"gamma": config["gamma"]},
        epochs=epochs,
        batch_size=batch_size,
        seed=seed,
    )

    trainer.fit(Xs, yc)
    prediction = trainer.predict(Xs) + ym

    return {
        "trainer": trainer,
        "x_mean": xm,
        "x_std": xs,
        "y_mean": ym,
        "config": config,
        "train_mse": float(MSE(y_train, prediction)),
        "train_r2": float(R2(y_train, prediction)),
    }


def fit_polynomials(
    x_train: Array,
    y_train: Array,
    settings: dict[str, tuple[int, float]],
) -> dict[str, Any]:
    """Refit the fixed Project 1 models.

    LLM-Assisted:
    ----
    See top of part_d.py for more details.

    """
    fitted = {}

    for name, (degree, lmbd) in settings.items():
        if name == "OLS":
            fitter = ModelFitter(OLS)
        elif name == "Ridge":
            fitter = ModelFitter(Ridge, {"lam": lmbd})
        else:
            fitter = ModelFitter(
                Lasso,
                {"lam": lmbd},
                optimizer=PolynomialAdam(),
                optimize_kwargs={"max_iter": 10000, "tol": 1e-8},
            )

        _, xm, xs, ym = fit_polynomial(fitter, x_train, y_train.ravel(), degree)
        prediction = predict_polynomial(fitter, x_train, degree, xm, xs, ym)

        fitted[name] = {
            "fitter": fitter,
            "degree": degree,
            "lambda": lmbd,
            "x_mean": xm,
            "x_std": xs,
            "y_mean": ym,
            "train_mse": float(MSE(y_train.ravel(), prediction)),
            "train_r2": float(R2(y_train.ravel(), prediction)),
        }

    return fitted


def fit_comparison(
    x_train: Array,
    y_train: Array,
    x_grid: Array,
    best: dict[str, Any],
    polynomial_settings: dict[str, tuple[int, float]],
    epochs: int = 500,
    batch_size: int = 16,
    seed: int = 2026,
) -> dict[str, Any]:
    """Fit fixed choices and return curves and training scores.

    LLM-Assisted:
    ----
    See top of part_d.py for more details.

    Fitted models are retained for the final test evaluation.
    """
    network_fits = {
        penalty: fit_selected(
            x_train, y_train, config, epochs=epochs, batch_size=batch_size, seed=seed
        )
        for penalty, config in best.items()
    }

    polynomial_fits = fit_polynomials(x_train, y_train, polynomial_settings)

    # Both networks use the same training input scaling.
    first_fit = next(iter(network_fits.values()))
    X_grid = (x_grid - first_fit["x_mean"]) / first_fit["x_std"]

    models = {}

    for penalty, fit in network_fits.items():
        prediction = fit["trainer"].predict(X_grid) + fit["y_mean"]

        models[f"NN ({penalty})"] = {
            "grid_prediction": prediction.ravel(),
            "MSE": fit["train_mse"],
            "train_r2": fit["train_r2"],
            "n_parameters": fit["config"]["n_parameters"],
        }

    for name, fit in polynomial_fits.items():
        prediction = predict_polynomial(
            fit["fitter"], x_grid, fit["degree"], fit["x_mean"], fit["x_std"], fit["y_mean"]
        )

        models[name] = {
            "grid_prediction": prediction.ravel(),
            "MSE": fit["train_mse"],
            "train_r2": fit["train_r2"],
            "n_parameters": fit["degree"] + 1,
        }

    return {
        "models": models,
        "network_fits": network_fits,
        "polynomial_fits": polynomial_fits,
    }


def final_comparison(
    x_test: Array,
    y_test: Array,
    comparison: dict[str, Any],
    project1_cv: dict[str, tuple[float, float]],
) -> list[dict[str, Any]]:
    """Test existing fits without refitting or repeating CV.

    LLM-Assisted:
    ----
    See top of part_d.py for more details.
    """
    target = y_test.ravel()
    network_fits = comparison["network_fits"]
    polynomial_fits = comparison["polynomial_fits"]

    first_fit = next(iter(network_fits.values()))
    X_test = (x_test - first_fit["x_mean"]) / first_fit["x_std"]

    rows = []

    for penalty, fit in network_fits.items():
        config = fit["config"]
        prediction = (fit["trainer"].predict(X_test) + fit["y_mean"]).ravel()

        rows.append(
            {
                "Model": f"NN ({penalty})",
                "Parameters": config["n_parameters"],
                "CV MSE": config["cv_mse"],
                "CV SE": config["cv_se"],
                "Test MSE": float(MSE(target, prediction)),
                "Test R2": float(R2(target, prediction)),
            }
        )

    for name, fit in polynomial_fits.items():
        prediction = predict_polynomial(
            fit["fitter"],
            x_test,
            fit["degree"],
            fit["x_mean"],
            fit["x_std"],
            fit["y_mean"],
        ).ravel()

        mse, se = project1_cv[name]

        rows.append(
            {
                "Model": name,
                "Parameters": fit["degree"] + 1,
                "CV MSE": mse,
                "CV SE": se,
                "Test MSE": float(MSE(target, prediction)),
                "Test R2": float(R2(target, prediction)),
            }
        )

    return rows
