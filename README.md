## `nns` contains small neural network implementations

Last midsummer I and a couple of my friends ended up having a quick chat about
machine learning. I know basically nothing about the subject but proceeded
to say something along the lines of "You could probably implement a simple neural
network from scratch in python in ~100 lines of code.". Saying this was probably
quite ignorant now that I think of it. This repository is meant for me to learn
the basics of neural networks and build up increasingly complex nets from scratch.

## Implementations

1. [hello_world.py](./hello_world.py) is the simplest thing I could come up that could be classifie as machine learning.
   Extensively documented and this nice for learning purposes. (64 LOC which consists mostly of comments.)
2. [gates_dumb.py](./gates_dumb.py) is a single neuron model for AND and OR gates. Uses finite differences
   instead of derivates/gradients. (~100 LOC)
4. [xor_dumb.py](./xor_dumb.py) is similar to `gates_dumb` except that is uses a more complex model
   so that it can model the XOR gate.
4. [xor_scalar.py](./xor_scalar.py) is our first multi layer neural network using `scalar framework`. (~50 LOC)

## Frameworks

1. [scalar.py](./scalar.py) is a very inefficient neural network library + a way to visualize
   expressions via `graphviz`. (~200 LOC)

## Ideas for future implementations

- [ ] XOR gate without scala framework?
