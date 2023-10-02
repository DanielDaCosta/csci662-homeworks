from model import Model
from nn_layers import FeedForwardNetwork


class NeuralModel(Model):
    def __init__(self): 
        self.network = FeedForwardNetwork()
        # might want to save other things here
        pass
    
    def train(self, input_file):
        pass

    def classify(self, input_file):
        pass