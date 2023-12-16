import random

import numpy as np
import copy


def tanh(x, derivative=False):
    if derivative:
        return 1 - np.tanh(x) ** 2
    else:
        return np.tanh(x)

def relu(x, derivative=False):
    if derivative:
        return np.where(x > 0, 1, 0)
    else:
        return np.maximum(0, x)


class CustomNetwork:

    def __init__(self, layers, activation):
        layers = list(layers)
        layers[0] += 1

        self.layers = layers
        self.activation = activation

        # Initialize the weights for each layer
        self.weights = [[]]
        for i in range(1, len(self.layers)):
            weight_matrix = np.random.randn(self.layers[i], self.layers[i-1])
            self.weights.append(weight_matrix)

        # Create outputs for each layer
        self.outputs = []
        for i in range(0, len(layers)):
            lst = [0] * layers[i]
            self.outputs.append(lst)

        pass

    def forward(self, input_data):
        input_data = list(input_data)
        input_data.append(1)

        # Calc
        self.outputs[0] = np.array(input_data)

        for i in range(1, len(self.layers)):
            k = np.dot(self.outputs[i-1], self.weights[i].T)
            l = self.activation(k)
            self.outputs[i] = l

        return self.outputs[-1]

    def calculate_error(self, target_output):
        # Mean Squared Error
        return np.mean((self.outputs[-1] - target_output) ** 2)

    def backpropagate(self, target_output, learning_rate=0.1):
        # Calculate the error
        error = self.outputs[-1] - target_output

        # Backpropagate the error
        for i in reversed(range(1, len(self.layers))):
            a = self.outputs[i].T
            b = self.activation(a, derivative=True)
            delta = error * b
            error = np.dot(delta, self.weights[i])

            _delta = np.expand_dims(delta, axis=-1)
            _outputs = np.expand_dims(self.outputs[i-1], axis=-1)

            d = learning_rate * np.dot(_delta, _outputs.T)
            self.weights[i] -= d
            pass

    def train(self, num_iterations, data):
        i = 0

        for _ in range(num_iterations):
            i += 1

            if i % 10 == 0:
                print()

            for inp, oup in data:
                # Forward pass
                o = self.forward(inp)

                # Calculate error
                err = self.calculate_error(oup)

                if i % 10 == 0:
                    print(inp, oup, o, err)

                # Backpropagation
                self.backpropagate(oup)


if __name__ == '__main__':
    print()
    print('WRONG FILE :)')
