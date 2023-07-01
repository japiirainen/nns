"Super simple and **dumb** single layer neural network for AND and OR gates."

import math
import random

from functools import reduce

AND_TRAIN = [
    # 0 && 0 == 0
    [0, 0, 0],
    # 1 && 0 == 0
    [1, 0, 0],
    # 0 && 1 == 0
    [0, 1, 0],
    # 1 && 1 == 1
    [1, 1, 1],
]

OR_TRAIN = [
    # 0 || 0 == 0
    [0, 0, 0],
    # 1 || 0 == 1
    [1, 0, 1],
    # 0 || 1 == 1
    [0, 1, 1],
    # 1 || 1 == 1
    [1, 1, 1],
]

TRAIN = OR_TRAIN

X = [x[:2] for x in TRAIN]
Y = [x[2] for x in TRAIN]

TRAIN_COUNT = len(TRAIN)


def sigmoid(x):
    """
    Sigmoid function. Maps any real number to [0, 1].
    """
    return 1 / (1 + math.exp(-x))


def rand_vec(n):
    """
    Returns a vector of length `n` contained of random real numbers.
    """
    return [random.random() for _ in range(n)]


def v_dot(xs, ys):
    """
    Returns the dot product of two vectors.
    """
    return sum(x * y for x, y in zip(xs, ys))


def cost(w, b):
    """
    Returns the cost of the model with regards to
    current weights and biases.
    """
    return (
        reduce(
            lambda acc, i: acc + (sigmoid(v_dot(w, X[i]) + b) - Y[i]) ** 2,
            range(TRAIN_COUNT),
            0,
        )
        / TRAIN_COUNT
    )


def dcost(eps, w, b):
    """
    Returns the cost of the model with regards to
    current weights and biases after shifting by the
    given epsilon.

    Here we are using the finite difference method for simplicity.
    In practice, we would would be calculating derivatives.
    """
    c = cost(w, b)

    def add_eps_at(i, x):
        return [a + eps if i == j else a for j, a in enumerate(x)]

    dws = [(cost(add_eps_at(i, w), b) - c) / eps for i in range(len(w))]
    db = (cost(w, b + eps) - c) / eps
    return dws, db


if __name__ == "__main__":
    w = rand_vec(2)
    b = random.random()

    RATE = 0.3
    EPOCH = 1000

    print(f"initial cost: {cost(w, b)}")

    for i in range(10 * EPOCH):
        dws, db = dcost(1e-1, w, b)
        dws = [RATE * dw for dw in dws]
        w = [a - b for a, b in zip(w, dws)]
        b -= RATE * db

        if i % EPOCH == 0:
            print(f"cost after {i} iterations: {cost(w, b)}")

    for i in range(2):
        for j in range(2):
            print(
                f"{('AND' if TRAIN == AND_TRAIN else 'OR')}({i}, {j}) = {sigmoid(w[0] * i + w[1] * j + b)}"
            )
