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
2. [dumb_gates.py](./dumb_gates.py) is a single layer net for AND and OR gates. Uses finite differences
   instead of derivates/gradients. (~100 LOC)

## Ideas for future implementations

- [ ] XOR gate. A single layer net no longer cuts it. Deep Learning here we come :-D
