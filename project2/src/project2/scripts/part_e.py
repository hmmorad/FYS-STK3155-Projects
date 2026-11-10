"""Part e: Classification on MNIST


Tools used:
    - Copilot autocomplete
    - ChatGPT (November 2026)


LLM-Assisted:

------------
Tool: ChatGPT (November 2026)
Role: Helped with skeleton code for part e.
Modifications: Fitted our implementation. Typing, and adjusted for our setup.
Promt: Explained what figures we wanted to plot, and asked it to give skeleton code for part e.

"""

from time import perf_counter
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression  # type: ignore
from sklearn.metrics import accuracy_score, f1_score  # type: ignore
from sklearn.neural_network import MLPClassifier  # type: ignore

from project2.network.activations import LReLU, ReLU, Sigmoid, Softmax
from project2.optimizers.adam import Adam
from project2.training.network_trainer import NetworkTrainer
from project2.typing import Array
from project2.utils.metrics import sd_error_binomial

ACTIVATIONS = {"Sigmoid": Sigmoid, "ReLU": ReLU, "Leaky ReLU": LReLU}


def _fit(
    X: Array,
    y: Array,
    config: dict[str, Any],
    epochs: int,
    batch_size: int,
    seed: int,
    X_val: Array | None = None,
    y_val: Array | None = None,
) -> tuple[NetworkTrainer, dict[str, list[float]], float]:
    """Fit a neural network with the given configuration,
    return the trainer, training history, and elapsed time.

    LLM-Assisted:
    ----
    See top of part_e.py for more details.
    """
    activation = ACTIVATIONS[config["activation"]]

    trainer = NetworkTrainer(
        network_kwargs={
            "n_inputs": X.shape[1],
            "layers": (
                tuple((nodes, activation) for nodes in config["hidden_sizes"])
                + ((y.shape[1], Softmax),)
            ),
            "lmbd1": config.get("lmbd1", 0.0),
            "lmbd2": config.get("lmbd2", 0.0),
            "seed": seed,
        },
        optimizer=Adam,
        optimizer_kwargs={"gamma": config["gamma"]},
        epochs=epochs,
        batch_size=batch_size,
        seed=seed,
    )

    # Match the float32 MNIST inputs
    for layer in trainer.network.layers:
        layer.W = layer.W.astype(np.float32)
        layer.b = layer.b.astype(np.float32)

    start = perf_counter()
    history = trainer.fit(X, y, X_val, y_val)

    return trainer, history, perf_counter() - start


def run_search(
    X: Array,
    y: Array,
    X_val: Array | None,
    y_val: Array | None,
    configs: list[dict[str, Any]],
    epochs: int = 10,
    batch_size: int = 256,
    seed: int = 2026,
) -> list[dict[str, Any]]:
    """Train supplied configurations on one fixed validation split.

    LLM-Assisted:
    ----
    See top of part_e.py for more details.

    """
    results = []

    for config in configs:
        config = {"lmbd1": 0.0, "lmbd2": 0.0, **config}
        trainer, history, seconds = _fit(X, y, config, epochs, batch_size, seed, X_val, y_val)
        score = history["val_accuracy"][-1]

        results.append(
            {
                "config": dict(config),
                "epochs": epochs,
                "parameters": sum(layer.W.size + layer.b.size for layer in trainer.network.layers),
                "validation_accuracy": score,
                "validation_se": sd_error_binomial(score, len(y_val)),
                "history": history,
                "fit_seconds": seconds,
            }
        )
    return results


def select_model(results: list[dict[str, Any]], n_validation: int) -> dict[str, Any]:
    """Select configuration and epoch jointly using one SE.

    LLM-Assisted:
    ----
    See top of part_e.py for more details.

    """
    candidates = []
    for result in results:
        history = result["history"]
        for epoch, (score, loss) in enumerate(
            zip(history["val_accuracy"], history["val_loss"], strict=True),
            1,
        ):
            if np.isfinite(score) and np.isfinite(loss):
                candidates.append(
                    {
                        **result,
                        "epochs": epoch,
                        "validation_accuracy": score,
                        "validation_se": sd_error_binomial(score, n_validation),
                    }
                )

    best = max(candidates, key=lambda row: row["validation_accuracy"])
    threshold = best["validation_accuracy"] - best["validation_se"]
    within = [row for row in candidates if row["validation_accuracy"] >= threshold]

    return min(
        within,
        key=lambda row: (
            row["parameters"],
            len(row["config"]["hidden_sizes"]),
            row["epochs"],
            -row["validation_accuracy"],
        ),
    )


def _fit_mlp(
    X: Array,
    labels: Array,
    config: dict[str, Any],
    epochs: int,
    batch_size: int,
    seed: int,
    X_val: Array | None = None,
    labels_val: Array | None = None,
) -> tuple[MLPClassifier, list[float]]:
    """Match supported activations and the custom L2 convention.

    LLM-Assisted:
    ----
    See top of part_e.py for more details.
    """
    if config["activation"] not in ("ReLU", "Sigmoid") or config.get("lmbd1", 0.0):
        raise ValueError("MLPClassifier supports ReLU/Sigmoid and L2, but not L1/Leaky ReLU.")

    model = MLPClassifier(
        hidden_layer_sizes=config["hidden_sizes"],
        activation=("relu" if config["activation"] == "ReLU" else "logistic"),
        solver="adam",
        learning_rate_init=config["gamma"],
        shuffle=False,
        random_state=seed,
    )

    rng = np.random.default_rng(seed)
    history = []

    for _ in range(epochs):
        order = rng.permutation(len(X))

        for start in range(0, len(X), batch_size):
            index = order[start : start + batch_size]
            model.batch_size = len(index)

            # sklearn's batch penalty is alpha/(2*n_batch) * ||W||².
            model.alpha = 2 * len(index) * config.get("lmbd2", 0.0)
            model.partial_fit(X[index], labels[index], classes=np.arange(10))

        if X_val is not None:
            history.append(accuracy_score(labels_val, model.predict(X_val)))

    return model, history


def fit_references(
    X: Array,
    y: Array,
    X_val: Array,
    y_val: Array,
    selected_ffnn: dict[str, Any],
    epochs: int = 30,
    batch_size: int = 1064,
    seed: int = 2026,
) -> dict[str, Any]:
    """Validate library models using the selected FFNN epoch budget.
    LLM-Assisted:
    ----
    See top of part_e.py for more details.

    """
    config = dict(selected_ffnn["config"])

    matched = config["activation"] in ("ReLU", "Sigmoid") and config.get("lmbd1", 0.0) == 0

    if not matched:
        config.update(activation="ReLU", lmbd1=0.0, lmbd2=0.0)

    mlp_epochs = selected_ffnn["epochs"]
    labels = y.argmax(axis=1)
    labels_val = y_val.argmax(axis=1)

    logistic = LogisticRegression(
        C=np.inf, solver="saga", max_iter=1000, tol=1e-3, random_state=seed
    )
    logistic.fit(X, labels)

    logistic_accuracy = float(accuracy_score(labels_val, logistic.predict(X_val)))

    _, history = _fit_mlp(X, labels, config, epochs, batch_size, seed, X_val, labels_val)

    return {
        "logistic_validation_accuracy": logistic_accuracy,
        "mlp_config": config,
        "mlp_history": history,
        "mlp_epochs": mlp_epochs,
        "mlp_matched": matched,
        "mlp_validation_accuracy": history[mlp_epochs - 1],
    }


def _record(
    name: str,
    config: dict[str, Any],
    parameters: int,
    seconds: float,
    labels: Array,
    prediction: Array,
) -> dict[str, Any]:
    """Record model evaluation metrics.

    LLM-Assisted:
    ----
    See top of part_e.py for more details.
    """
    return {
        "Model": name,
        "Architecture": str(config["hidden_sizes"]),
        "Parameters": parameters,
        "Fit time (s)": seconds,
        "Test accuracy": float(accuracy_score(labels, prediction)),
        "Macro F1": float(
            f1_score(
                labels,
                prediction,
                average="macro",
                zero_division=0,
            )
        ),
    }


def final_comparison(
    X: Array,
    y: Array,
    X_test: Array,
    labels_test: Array,
    selected: dict[str, Any],
    references: dict[str, Any],
    batch_size: int = 1064,
    seed: int = 2026,
) -> dict[str, Any]:
    """Refit on all training images and evaluate fixed models once.
    LLM-Assisted:
    ----
    See top of part_e.py for more details.
    """
    records = []
    predictions = {}

    for name, choice in selected.items():
        config = choice["config"]

        trainer, _, seconds = _fit(X, y, config, choice["epochs"], batch_size, seed)
        pred = trainer.predict(X_test).argmax(axis=1)
        predictions[name] = pred
        records.append(_record(name, config, choice["parameters"], seconds, labels_test, pred))

    labels = y.argmax(axis=1)
    start = perf_counter()

    model = LogisticRegression(C=np.inf, solver="saga", max_iter=1000, tol=1e-3, random_state=seed)
    model.fit(X, labels)
    seconds = perf_counter() - start

    pred = model.predict(X_test)
    name = "Scikit-Learn logistic"
    predictions[name] = pred

    records.append(
        _record(
            name,
            {"hidden_sizes": ()},
            int(model.coef_.size + model.intercept_.size),
            seconds,
            labels_test,
            pred,
        )
    )

    config = references["mlp_config"]
    start = perf_counter()

    model, _ = _fit_mlp(
        X,
        labels,
        config,
        references["mlp_epochs"],
        batch_size,
        seed,
    )
    seconds = perf_counter() - start

    pred = model.predict(X_test)
    name = "Scikit-Learn MLP" if references["mlp_matched"] else "Scikit-Learn MLP reference"
    predictions[name] = pred

    parameters = sum(
        weights.size + biases.size
        for weights, biases in zip(
            model.coefs_,
            model.intercepts_,
            strict=True,
        )
    )

    records.append(_record(name, config, int(parameters), seconds, labels_test, pred))

    return {"records": records, "predictions": predictions}
