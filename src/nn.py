from graphviz import Digraph
from random import random
import engine

if __name__ == '__main__':
    a = engine.Value(random())
    b = engine.Value(random())
    c = engine.Value(random())
    d = a * b + c
    d = d.tanh()
    d.backward()
