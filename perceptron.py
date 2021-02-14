import numpy as np

class Perceptron:
    weights = []
    learning_rate = 0.5
    
    def __init__(self, num_inputs, learning_rate=0.5):
        self.weights = np.ones(num_inputs + 1)
        self.learning_rate = learning_rate
        
    def wonder(self, data):
        data = np.insert(data, 0, 1)
        
        dot = np.dot(self.weights, data)
        return 1 if dot >= 0 else 0
    
    def perceive(self, data, sol):
        est = self.wonder(data)
        data = np.insert(data, 0, 1)
        
        self.weights += self.learning_rate * (sol - est) * data