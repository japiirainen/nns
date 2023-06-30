"Super simple and **dumb** neural network for solving AND and OR gates."

import math
import random

AND_TRAIN = [
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [1, 1, 1],
]

OR_TRAIN = [
    [0, 0, 0],
    [1, 0, 1],
    [0, 1, 1],
    [1, 1, 1],
]

TRAIN = OR_TRAIN

TRAIN_COUNT = len(TRAIN)


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def rand_vec(n):
    return [random.random() for _ in range(n)]


def dot(xs, ys):
    return sum(x * y for x, y in zip(xs, ys))


def cost(w, b):
    r = 0
    for i in range(TRAIN_COUNT):
        x = TRAIN[i][:2]
        y = TRAIN[i][2]
        x = dot(w, x) + b
        x = sigmoid(x)
        r += math.pow(x - y, 2)
    return r / TRAIN_COUNT


def dcost(eps, w, b):
    c = cost(w, b)

    def add_eps_at(i, x):
        return [a + eps if i == j else a for j, a in enumerate(x)]

    dws = [(cost(add_eps_at(i, w), b) - c) / eps for i in range(len(w))]
    db = (cost(w, b + eps) - c) / eps
    return dws, db


def vec_sub(x, y):
    return [a - b for a, b in zip(x, y)]


if __name__ == "__main__":
    w = rand_vec(2)
    b = random.random()

    RATE = 0.3
    EPOCH = 1000

    print(cost(w, b))

    for _ in range(10 * EPOCH):
        c = cost(w, b)
        eps = 1e-1
        dws, db = dcost(eps, w, b)
        w = vec_sub(w, [RATE * dw for dw in dws])
        b -= RATE * db

    for i in range(2):
        for j in range(2):
            print(
                f"{('AND' if TRAIN == AND_TRAIN else 'OR')}({i}, {j}) = {sigmoid(w[0] * i + w[1] * j + b)}"
            )
