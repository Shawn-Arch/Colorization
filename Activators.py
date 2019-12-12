import numpy as np


class NoneActivator(object):
    def forward(self, weighted_input):
        #return weighted_input
        return weighted_input

    def backward(self, output):
        return 1

class ReluActivator(object):
    def forward(self, weighted_input):
        #return weighted_input
        return weighted_input if weighted_input > 0 else weighted_input * 0.25

    def backward(self, output):
        return 1 if output > 0 else 0.25


class IdentityActivator(object):
    def forward(self, weighted_input):
        return weighted_input

    def backward(self, output):
        return 1


class SigmoidActivator(object):
    def forward(self, weighted_input):
        return 1.0 / (1.0 + np.exp(-weighted_input))
    #the partial of sigmoid
    def backward(self, output):
        return output * (1 - output)


class TanhActivator(object):
    def forward(self, weighted_input):
        return 2.0 / (1.0 + np.exp(-2 * weighted_input)) - 1.0

    def backward(self, output):
        return 1 - output * output
