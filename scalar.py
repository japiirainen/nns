"""
scalar.py

Neural Network framework built up from scratch. Uses `Scalar` class as it's
fundamental building block. `Scalar` class is a wrapper around a float value
that keeps track of its gradient.

References:
- karpathy/micrograd

"""

import random
import graphviz as gv
import numpy as np  # for exp since math.exp overflows easily


def build_topo_order(node):
    path = []
    vis = set()

    def go(v):
        if v not in vis:
            vis.add(v)
            for kid in v._kids:
                go(kid)
            path.append(v)

    go(node)

    return path


class Scalar:
    def __init__(self, value, kids=(), label=None):
        self.value = value
        self._kids = kids
        self.grad = 0.0
        self._backward = lambda: None
        self.op = None
        self.label = label

    def backward(self):
        self.grad = 1.0

        for node in reversed(build_topo_order(self)):
            node._backward()

    def tanh(self):
        t = (np.exp(2 * self.value) - 1) / (np.exp(2 * self.value) + 1)
        s = Scalar(t, (self,))
        s.op = "tanh"
        s.label = f"tanh({self.label})"

        def _tanhb():
            self.grad += (1 - t * t) * s.grad

        s._backward = _tanhb

        return s

    def __add__(self, other):
        if isinstance(other, int):
            other = Scalar(other)

        s = Scalar(self.value + other.value, (self, other))
        s.op = "+"
        s.label = f"{self.label} + {other.label}"

        def _addb():
            self.grad += s.grad
            other.grad += s.grad

        s._backward = _addb

        return s

    def __mul__(self, other):
        if isinstance(other, int):
            other = Scalar(other)

        s = Scalar(self.value * other.value, (self, other))
        s.op = "*"
        s.label = f"{self.label} * {other.label}"

        def _mulb():
            self.grad += other.value * s.grad
            other.grad += self.value * s.grad

        s._backward = _mulb

        return s

    def __pow__(self, other):
        assert isinstance(other, int)

        s = Scalar(self.value**other, (self,))

        s.op = "**"
        s.label = f"{self.label} ** {other}"

        def _powb():
            self.grad += other * self.value ** (other - 1) * s.grad

        s._backward = _powb

        return s

    def __neg__(self):
        return self * (-1)

    def __sub__(self, other):
        return self + (-other)

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * other**-1

    def __rtruediv__(self, other):
        return other * self**-1

    def graph(self, name):
        dot = gv.Digraph(comment=name)

        def go(v):
            op_key = f"{v.label} {v.op}"
            for kid in v._kids:
                go(kid)
                dot.edge(kid.label, op_key)

            if v.op:
                dot.node(op_key, label=v.op)
                dot.edge(op_key, v.label)

            dot.node(
                v.label,
                f"{v.label}\nvalue = {v.value:.4f}\ngrad = {v.grad:.4f}",
                shape="box",
            )

        go(self)

        dot.render(name, view=False)

    def __repr__(self):
        return f"Scalar({self.value:.4f})"


# Neural nets


class Neuron:
    def __init__(self, n_weights, non_linear=True):
        self.w = [Scalar(random.uniform(-1, 1)) for _ in range(n_weights)]
        self.b = Scalar(0.0)
        self.non_linear = non_linear

    def parameters(self):
        return self.w + [self.b]

    def forward(self, x):
        n = sum(wi * xi for wi, xi in zip(self.w, x)) + self.b
        return n.tanh() if self.non_linear else n


class Layer:
    def __init__(self, n_inputs, n_outputs, **kwargs):
        self.neurons = [Neuron(n_inputs, **kwargs) for _ in range(n_outputs)]

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def forward(self, x):
        xs = [n.forward(x) for n in self.neurons]
        return xs[0] if len(xs) == 1 else xs


class MultiLayerPerceptron:
    def __init__(self, n_inputs, n_outputs):
        ns = [n_inputs] + n_outputs
        # We want the last layer (output) to be linear
        non_linear = lambda i: i != len(n_outputs) - 1
        self.layers = [
            Layer(ns[i], ns[i + 1], non_linear=non_linear(i))
            for i in range(len(n_outputs))
        ]

    def parameters(self):
        return [p for l in self.layers for p in l.parameters()]

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0
