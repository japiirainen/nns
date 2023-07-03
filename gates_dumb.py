"""
In this file we model AND and OR gates using a single neuron. We do things
very manually without using matrices or gradient descent.

For these simple gates we can use a single neuron 'network'.

 |---|
 | a |\
 |---| \    |---|     |---|
        --> | c | --> | o |
 |---| /    |---|     |---|
 | b |/
 |---|

"""

import math
import random

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

# Change this or `AND_TRAIN` to test AND gate
TRAIN = OR_TRAIN

X = [x[:2] for x in TRAIN]
Y = [x[2] for x in TRAIN]

TRAIN_COUNT = len(TRAIN)


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


# In `forward` we compute the output of the model for a given input.
def forward(w1, w2, x1, x2, b):
    return sigmoid(x1 * w1 + x2 * w2 + b)


# In `loss` we compute the mean squared error of the model on the training data.
def loss(w1, w2, b):
    out = 0.0

    for i in range(TRAIN_COUNT):
        out += (forward(w1, w2, X[i][0], X[i][1], b) - Y[i]) ** 2

    return out / TRAIN_COUNT


# In `dloss` we approximate the derivative of the loss function with respect to
# each weight and bias by using finite differences.
def dloss(eps, w1, w2, b):
    l = loss(w1, w1, b)

    dw1 = (loss(w1 + eps, w2, b) - l) / eps
    dw2 = (loss(w1, w2 + eps, b) - l) / eps
    db = (loss(w1, w2, b + eps) - l) / eps

    return dw1, dw2, db


if __name__ == "__main__":
    # Initialize model with random weights and biases
    w1 = random.random()
    w2 = random.random()
    b = random.random()

    print(f"initial weights: {w1}, {w2}, {b}")

    RATE = 0.1
    EPOCH = 1000

    print(f"initial loss: {loss(w1, w2, b)}")

    for i in range(10 * EPOCH):
        EPS = 0.1
        dw1, dw2, db = dloss(EPS, w1, w2, b)
        # Update weights and biases.
        # This is where the so called 'learning' happens.
        w1 -= RATE * dw1
        w2 -= RATE * dw2
        b -= RATE * db

        if i % EPOCH == 0:
            print(f"loss after {i} iterations: {loss(w1, w2, b)}")

    for i in range(2):
        for j in range(2):
            print(
                f"{('AND' if TRAIN == AND_TRAIN else 'OR')}({i}, {j}) = {sigmoid(w1 * i + w2 * j + b)}"
            )
