"""Part c: activation, depth and width with Adam.

Tools used:
    - Copilot autocomplete
    - ChatGPT (November 2026)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from project2.network.activations import Identity, LReLU, ReLU, Sigmoid
from project2.optimizers.adam import Adam
from project2.training.cross_validation import CrossValidation

if TYPE_CHECKING:
    from typing import Any

    from project2.typing import Array

ACTIVATIONS = {"Sigmoid": Sigmoid, "ReLU": ReLU, "Leaky ReLU": LReLU}


def run(
    x_train: Array,
    y_train: Array,
    depths: list[int],
    widths: list[int],
    gamma: float = 0.01,
    epochs: int = 1000,
    batch_size: int = 16,
    n_splits: int = 5,
    seed: int = 2026,
) -> dict[str, Any]:
    """Compare activation, depth and width at a fixed learning rate.
    Args:
        x_train: Training input data.
        y_train: Training target data.
        depths: List of network depths to evaluate.
        widths: List of network widths to evaluate.
        gamma: Learning rate for the Adam optimizer.
        epochs: Number of training epochs.
        batch_size: Size of each training batch.
        n_splits: Number of cross-validation splits.
        seed: Random seed for reproducibility.

    Returns:
        Dictionary containing cross-validation results, training MSE, and selected model details.
    """
    shape = (len(ACTIVATIONS), len(depths), len(widths))

    cv_mse = np.full(shape, np.inf)
    cv_se = np.full(shape, np.inf)
    train_mse = np.full(shape, np.nan)

    for a, (_, activation) in enumerate(ACTIVATIONS.items()):
        for d, depth in enumerate(depths):
            for w, width in enumerate(widths):
                hidden_sizes = (int(width),) * int(depth)
                cv = CrossValidation(
                    network_kwargs={
                        "n_inputs": x_train.shape[1],
                        "layers": (
                            tuple((nodes, activation) for nodes in hidden_sizes) + ((1, Identity),)
                        ),
                        "seed": seed,
                    },
                    optimizer=Adam,
                    optimizer_kwargs={"gamma": gamma},
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

                # Our regression loss is half-MSE.
                mse = 2.0 * result.mean_score
                se = 2.0 * result.standard_error

                if np.isfinite(mse) and np.isfinite(se):
                    cv_mse[a, d, w] = mse
                    cv_se[a, d, w] = se
                    train_mse[a, d, w] = 2.0 * result.mean_train_score

    return select_models(
        {"cv_mse": cv_mse, "cv_se": cv_se, "train_mse": train_mse},
        depths,
        widths,
        n_splits=n_splits,
        n_inputs=x_train.shape[1],
        gamma=gamma,
    )


def select_models(
    results: dict[str, Any],
    depths: list[int],
    widths: list[int],
    n_splits: int = 5,
    n_inputs: int = 1,
    gamma: float = 0.01,
) -> dict[str, Any]:
    """Apply one-SE selection separately to each activation.

    LLM-assisted
    ------------
    Tool: ChatGPT (November 2026)
    Role: Implemented the one-standard-error model selection procedure for each activation.
    Modifications: To fit our implementation overall.
    Prompt: Gave it our code and asked for one-standard-error model selection.

    """
    cv_mse = results["cv_mse"]
    cv_se = results["cv_se"]
    parameters = np.empty((len(depths), len(widths)), dtype=int)

    for d, depth in enumerate(depths):
        for w, width in enumerate(widths):
            hidden_sizes = (int(width),) * int(depth)
            sizes = (n_inputs, *hidden_sizes, 1)

            parameters[d, w] = sum(
                (n_in + 1) * n_out for n_in, n_out in zip(sizes[:-1], sizes[1:], strict=True)
            )

    selections = {}
    selected_models = {}

    for a, name in enumerate(ACTIVATIONS):
        means = cv_mse[a]
        errors = cv_se[a]
        valid = np.isfinite(means) & np.isfinite(errors)

        if not valid.any():
            raise ValueError(f"No finite CV MSE and SE for {name}.")

        minimum = tuple(
            int(i) for i in np.unravel_index(np.argmin(np.where(valid, means, np.inf)), means.shape)
        )

        threshold = float(means[minimum] + errors[minimum])
        within = valid & (means <= np.nextafter(threshold, np.inf))
        candidates = [tuple(int(i) for i in index) for index in np.argwhere(within)]

        chosen = min(
            candidates,
            key=lambda i: (parameters[i], depths[i[0]], means[i]),
        )
        selections[name] = {
            "minimum": minimum,
            "within": within,
            "selected": chosen,
            "threshold": threshold,
        }

        d, w = chosen
        index = (a, d, w)
        selected_models[name] = {
            "activation": name,
            "hidden_sizes": (int(widths[w]),) * int(depths[d]),
            "gamma": gamma,
            "n_parameters": int(parameters[chosen]),
            "cv_mse": float(cv_mse[index]),
            "cv_se": float(cv_se[index]),
            "train_mse": float(results["train_mse"][index]),
        }

    return {
        **results,
        "selection": selections,
        "selected": selected_models,
    }
