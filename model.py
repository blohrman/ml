import numpy as np

class Model:
    model = None
    
    def __init__(self, model):
        self.model = model
        
    def train(self, data, sols, max_epochs=1000):
        epoch = 0
        loss = self.loss(data, sols)
        
        while (epoch != max_epochs and loss != 0):
            for row, sol in zip(data, sols):
                self.model.perceive(row, sol)
            
            epoch += 1
            
    def test(self, data):
        return self.model.wonder(data)
            
    def loss(self, data, sols):
        tot = 0
        for row, sol in zip(data, sols):
            pred = self.model.wonder(row)
            e = sol - pred
            tot += e ** 2
        
        coeff = 1 / (2 * len(data))
        return coeff * tot