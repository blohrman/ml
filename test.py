import numpy as np
from perceptron import Perceptron
from model import Model

test = Perceptron(2)
model = Model(test)

x = np.array([[0, 0],
     [0, 1],
     [1, 0],
     [1, 1]])
y = np.array([0, 1, 1, 1])

model.train(x, y)

print(model.model.weights)