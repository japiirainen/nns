"""
This is a simple example of a MultiLayerPerceptron that learns the XOR function.
In this example we use the scalar implementation of a simple neural network library.

Why XOR gate?

The XOR gate is a simple example of a function that is not linearly separable.
This means that it is not possible to draw a straight line that separates the
inputs that give 0 from the inputs that give 1. Because of this the problem
cannot be solved by a single perceptron, but it can be solved by a MultiLayerPerceptron.
"""

from scalar import MultiLayerPerceptron

X = [
    [0, 0],
    [1, 0],
    [0, 1],
    [1, 1],
]

Y = [0, 1, 1, 0]


if __name__ == "__main__":
    # We use model with 2 inputs, 2 hidden units and 1 output.
    #
    # |---|       |---|
    # | a | ----> | c |\
    # |---| \  /  |---| \    |---|
    #        \/          --> | o |
    # |---|  /\   |---| /    |---|
    # | b | /---> | d |/
    # |---|       |---|
    #
    model = MultiLayerPerceptron(2, [2, 1], activation="tanh")

    for i in range(1000):
        # Forward pass
        scores = [model.forward(xi) for xi in X]
        total_loss = sum(((si - yi) ** 2) for yi, si in zip(Y, scores))

        if total_loss.value < 1e-5:
            break

        # Backward pass
        model.zero_grad()
        total_loss.backward()

        # Update parameters
        LEARNING_RATE = 0.1
        for p in model.parameters():
            p.value -= LEARNING_RATE * p.grad

    for i in range(2):
        for j in range(2):
            print(f"{i} XOR {j} -> {model.forward([i, j]).value:.4f}")
