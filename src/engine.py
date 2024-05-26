import math


class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0
        self._prev = set(_children)
        self._op = _op

    def __repr__(self):
        return self.data

    def __add__(self, other):
        return Value(self.data + other.data, (self, other), '+')

    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), '*')
        # out.grad += other.grad
        return out

    def tanh(self):
        out = Value((math.exp(2 * self.data) - 1) / ((math.exp(2 * self.data)) + 1), (self,), 'tanh')
        return out
