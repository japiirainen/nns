"""
XOR gate modeled with a two layer neural network without any framework.

Why XOR gate?

The XOR gate is interesting because it is not linearly separable. This means
the model in ./gates_dumb.py cannot learn it. We need a more complex model.
We will use a two layer neural network. E.g.


 |---|       |---|
 | a | ----> | c |\
 |---| \  /  |---| \    |---|
        \/          --> | o |
 |---|  /\   |---| /    |---|
 | b | /---> | d |/
 |---|       |---|

"""

import math
import random
from dataclasses import dataclass

X = [
    [0, 0],
    [1, 0],
    [0, 1],
    [1, 1],
]

Y = [0, 1, 1, 0]


@dataclass
class Model:
    a_w1: float
    a_w2: float
    a_b1: float

    b_w1: float
    b_w2: float
    b_b1: float

    c_w1: float
    c_w2: float
    c_b1: float


def random_model():
    return Model(
        a_w1=random.random(),
        a_w2=random.random(),
        a_b1=random.random(),
        b_w1=random.random(),
        b_w2=random.random(),
        b_b1=random.random(),
        c_w1=random.random(),
        c_w2=random.random(),
        c_b1=random.random(),
    )


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


# In `forward` we compute the output of the model for a given input.
def forward(m, x1, x2):
    a = sigmoid(m.a_w1 * x1 + m.a_w2 * x2 + m.a_b1)
    b = sigmoid(m.b_w1 * x1 + m.b_w2 * x2 + m.b_b1)
    return sigmoid(m.c_w1 * a + m.c_w2 * b + m.c_b1)


def loss(m):
    out = 0.0
    for x, y in zip(X, Y):
        out += (forward(m, x[0], x[1]) - y) ** 2
    return out / len(X)


def finite_diff(m, eps):
    l = loss(m)

    s = m.a_w1
    m.a_w1 += eps
    da_w1 = (loss(m) - l) / eps
    m.a_w1 = s

    s = m.a_w2
    m.a_w2 += eps
    da_w2 = (loss(m) - l) / eps
    m.a_w2 = s

    s = m.a_b1
    m.a_b1 += eps
    da_b1 = (loss(m) - l) / eps
    m.a_b1 = s

    s = m.b_w1
    m.b_w1 += eps
    db_w1 = (loss(m) - l) / eps
    m.b_w1 = s

    s = m.b_w2
    m.b_w2 += eps
    db_w2 = (loss(m) - l) / eps
    m.b_w2 = s

    s = m.b_b1
    m.b_b1 += eps
    db_b1 = (loss(m) - l) / eps
    m.b_b1 = s

    s = m.c_w1
    m.c_w1 += eps
    dc_w1 = (loss(m) - l) / eps
    m.c_w1 = s

    s = m.c_w2
    m.c_w2 += eps
    dc_w2 = (loss(m) - l) / eps
    m.c_w2 = s

    s = m.c_b1
    m.c_b1 += eps
    dc_b1 = (loss(m) - l) / eps
    m.c_b1 = s

    return Model(
        a_w1=da_w1,
        a_w2=da_w2,
        a_b1=da_b1,
        b_w1=db_w1,
        b_w2=db_w2,
        b_b1=db_b1,
        c_w1=dc_w1,
        c_w2=dc_w2,
        c_b1=dc_b1,
    )


if __name__ == "__main__":
    # Initialize model with random weights and biases
    m = random_model()

    print(f"initial model: {m}")

    RATE = 0.1
    EPOCH = 1000

    print(f"initial loss: {loss(m)}")

    for i in range(100 * EPOCH):
        EPS = 0.1
        g = finite_diff(m, EPS)
        # Update weights and biases.
        # This is where the so called 'learning' happens.
        m.a_w1 -= RATE * g.a_w1
        m.a_w2 -= RATE * g.a_w2
        m.a_b1 -= RATE * g.a_b1
        m.b_w1 -= RATE * g.b_w1
        m.b_w2 -= RATE * g.b_w2
        m.b_b1 -= RATE * g.b_b1
        m.c_w1 -= RATE * g.c_w1
        m.c_w2 -= RATE * g.c_w2
        m.c_b1 -= RATE * g.c_b1

        if i % EPOCH == 0:
            print(f"loss after {i} iterations: {loss(m)}")

    for i in range(2):
        for j in range(2):
            print(f"{i} XOR {j} -> {forward(m, i, j)}")
