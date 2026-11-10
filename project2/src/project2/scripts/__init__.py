from project2.network.activations import LReLU, ReLU, Sigmoid

ACTIVATIONS = {"Sigmoid": Sigmoid, "ReLU": ReLU, "Leaky ReLU": LReLU}

__all__ = ["ACTIVATIONS"]
