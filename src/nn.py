import random
import engine


class Neuron:
    def __init__(self, inputs):
        self.w = [engine.Value(random.uniform(-1, 1)) for _ in range(inputs)]
        self.b = engine.Value(random.uniform(-1, 1))

    def __call__(self, x):
        activation = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        out = activation.tanh()
        return out

    def parameters(self):
        return self.w + [self.b]


class Layer:
    def __init__(self, inputs, outputs):
        self.neurons = [Neuron(inputs) for _ in range(outputs)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]


class MLP:
    def __init__(self, inputs, outputs):
        sz = [inputs] + outputs
        self.layers = [Layer(sz[i], sz[i + 1]) for i in range(len(outputs))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]


if __name__ == '__main__':
    xs = [
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0],
    ]

    ys = [1.0, -1.0, 1.0, 1.0]

    n = MLP(3, [4, 4, 1])

    # TRAINING LOOP
    for i in range(20):
        # forward pass
        ypred = [n(x) for x in xs]
        loss = sum((yout - ygt) ** 2 for ygt, yout in zip(ys, ypred))

        # backward pass
        loss.backward()

        # Update
        for p in n.parameters():
            p.data -= 0.05 * p.grad

        print(loss)
