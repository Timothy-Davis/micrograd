from graphviz import Digraph
from random import random
import engine

if __name__ == '__main__':
    a = engine.Value(2.0)
    b = engine.Value(0.0)
    c = engine.Value(-3.0)
    d = engine.Value(1.0)
    e = engine.Value(6.8813735870195432)
    f = a*c + b*d + e
    g = f.relu()
    print(g, f, e, d, c, b, a)
    g.backward()
    print(a.grad)
    print(b.grad)
    print(c.grad)
    print(d.grad)
