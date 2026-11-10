# FYS-STK3155 Project 2: Feed-Forward Neural Networks from Scratch

A feed-forward neural network implemented with NumPy, including dense layers, activation functions, loss functions and backpropagation from scratch. We apply it to Runge regression and handwritten digit classification with MNIST.

[Results notebook](notebooks/Project2_results.ipynb) · [Tests notebook](notebooks/Project2_tests.ipynb) · [Report PDF](report/main.pdf)

## Topics

- Forward propagation and backpropagation for fully connected networks.
- Sigmoid, ReLU and Leaky ReLU as activation functions.
- Linear outputs with MSE for regression and softmax outputs with cross-entropy for multiclass classification.
- Gradient descent, RMSProp and Adam, with full-batch and minibatch training.
- Network depth, width, learning rate and L1/L2 regularization.
- K-fold cross-validation and one-standard-error model selection.

## Project layout

```text
project2/
├── README.md
├── src/project2/
│   ├── network/
│   │   ├── FFNN.py                 # NeuralNetwork class
│   │   ├── layer.py                # DenseLayer class
│   │   ├── activations.py
│   │   └── losses.py
│   ├── optimizers/                 # Gradient descent, RMSProp and Adam
│   ├── training/
│   │   ├── network_trainer.py      # Training loop and learning histories
│   │   └── cross_validation.py     # Crossvalidation
│   ├── scripts/                    # Experiments: part_b.py to part_e.py
│   ├── utils/                      # Encoding, metrics and table export
│   ├── typing.py
│   └── py.typed
├── tests/                          # Backpropagation and cross-validation checks
├── notebooks/
│   ├── Project2_results.ipynb
│   └── Project2_tests.ipynb
├── datasets/
│   └── mnist.npz
├── figures/                        # Generated PDF figures
└── report/                         # LaTeX source, report PDF and tables
```

## Creating and training a network

A network is defined by its number of input features and a sequence of layers. Each layer is written as `(number_of_neurons, ActivationClass)`, including the output layer.

For example, the following creates a network with one input feature, one sigmoid hidden layer of 50 neurons and one linear output:

```python
from project2.network.activations import Identity, Sigmoid
from project2.optimizers.adam import Adam
from project2.training.network_trainer import NetworkTrainer

network_kwargs = {
    "n_inputs": 1,
    "layers": (
        (50, Sigmoid),   # Hidden layer
        (1, Identity),   # Linear output, paired with MSE
    ),
    "lmbd1": 0.0,
    "lmbd2": 0.0,
    "seed": 2026,
}

trainer = NetworkTrainer(
    network_kwargs=network_kwargs,
    optimizer=Adam,
    optimizer_kwargs={"gamma": 0.01},
    epochs=1000,
    batch_size=16,
    seed=2026,
)

history = trainer.fit(x_train, y_train)
predictions = trainer.predict(x_test)
```

This example assumes the data has already been prepared. For one-dimensional regression, inputs and targets have shape `(n_samples, 1)`.

Additional hidden layers are added to the `layers` tuple. The `lmbd1` and `lmbd2` settings control L1 and L2 regularization.

For MNIST, the network has 784 input features and an output layer `(10, Softmax)`, paired with multiclass cross-entropy. Training targets are one-hot encoded with shape `(n_samples, 10)`. Using only this output layer, with no hidden layers, gives multiclass logistic regression.

The implementation separates three responsibilities:

- [`DenseLayer`](src/project2/network/layer.py) stores the weights, biases and activation function for one layer, and calculates its forward pass and gradients.
- [`NeuralNetwork`](src/project2/network/FFNN.py) connects the layers and handles the full forward pass, backpropagation and regularized cost.
- [`NetworkTrainer`](src/project2/training/network_trainer.py) creates the network and optimizer, forms minibatches, updates parameters and records the training history.


## Experiments

| Script | Experiment |
| --- | --- |
| [`part_b.py`](src/project2/scripts/part_b.py) | Runge regression: compare the FFNN with OLS and Scikit-learn's MLP, then study optimizers and learning rates. |
| [`part_c.py`](src/project2/scripts/part_c.py) | Compare activation functions, network depth and width using cross-validation. |
| [`part_d.py`](src/project2/scripts/part_d.py) | Select L1/L2-regularized networks and compare final regression models on the test set. |
| [`part_e.py`](src/project2/scripts/part_e.py) | MNIST model selection, learning curves, library comparisons and final test evaluation. |

## Running the experiments

Follow the [shared setup instructions](../README.md#setup), running `uv sync` from the repository root. Select that environment's `.venv` kernel for both notebooks.

1. Run [Project2_tests.ipynb](notebooks/Project2_tests.ipynb) for the verification checks.
2. Run [Project2_results.ipynb](notebooks/Project2_results.ipynb) in order to reproduce the experiments, figures and tables.

Epoch counts, batch sizes, random seeds and search grids are configured in the results notebook. Figures are saved in `figures/`, and LaTeX tables in `report/tables/`. Set `FIGURES_DIR` or `TABLES_DIR` to `None` in the notebook to disable the corresponding exports.

### Data

The regression experiments reuse the Runge data and training/test split from Project 1. Project 2 imports its polynomial models and shared helpers from the `project1` package.

For classification, the notebook uses `datasets/mnist.npz` with arrays named `x_train`, `y_train`, `x_test` and `y_test`. It scales pixel values to `[0, 1]` and one-hot encodes the training targets. Model selection uses 50,000 training images and 10,000 validation images. Final models are refitted on all 60,000 training images and evaluated on the separate 10,000-image test set.

## Verification

The tests compare backpropagation gradients with JAX automatic differentiation and check the cross-validation implementation. They can also be run from the repository root:

```bash
uv run pytest project2/tests
```

The neural network is trained using our NumPy implementation. JAX is used for gradient checks, while we use Scikit-learn's `LogisticRegression` and `MLP` to compare. See the root README for the [shared libraries](../README.md#libraries).

## Use of AI tools

This README was drafted with ChatGPT in November 2026. Assistance with the implementation, plots and report is documented in the source files and report.