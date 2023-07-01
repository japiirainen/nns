"The simplest machine learning model I could come up with. Thus calling this the Hello World of ML."

import random
import sys

TRAIN = [
    [0, 0],
    [1, 3],
    [2, 6],
    [3, 9],
    [4, 12],
    [5, 15],
]

TRAIN_COUNT = len(TRAIN)

# After a bit of thought we can clearly see that the model is y = x*3.
# We must pretend we don't know this and let the model figure it out.


def loss(w):
    "The loss function is the sum of the squared errors."
    return sum((w * x - y) ** 2 for x, y in TRAIN) / TRAIN_COUNT


if __name__ == "__main__":
    # We will use a simple linear model, y = w * x
    # Initially w is a random real between 0 and 1.
    # Our model's job is to figure out the value of w that best fits the data.
    w = random.random()

    l = loss(w)
    print(f"Initial loss: {l}")

    # In real world we would use gradient descent to drive the value of w towards 0.
    # To avoid the math we will use finite differences,
    # which is a numerical approximation of the derivative.
    EPS = 0.001
    # rate is the learning rate, which is a hyperparameter that controls how fast the model learns.
    # without it the model will not converge.
    RATE = 0.001

    for i in range(sys.maxsize):
        if loss(w) < 0.0001:
            break

        # dl is the approximation of the derivative of the loss function.
        dl = (loss(w + EPS) - loss(w)) / EPS

        # adjust w by the dl. This will hopefully drive the loss down.
        w -= RATE * dl

        if i % 1000 == 0:
            print(f"Loss at iteration {i}: {loss(w)}")

    print("actual: f(x) = 3x")
    for x in range(TRAIN_COUNT):
        print(f"{x} -> {x * 3}")

    print("----------")

    print("our model's prediction: f(x) = wx")
    for x in range(TRAIN_COUNT):
        print(f"{x} -> {round(w * x, 2)}")
