import random

import numpy as np
import copy


def tanh(x, derivative=False):
    if derivative:
        return 1 - np.tanh(x) ** 2
    else:
        return np.tanh(x)


class CustomNetwork:

    def __init__(self, layers, interconnects, activation):
        self.layers = layers
        self.activation = activation

        # Initialize the weights for each layer
        self.weights = []
        for i in range(0, len(self.layers) - 1):
            weight_matrix = np.random.randn(self.layers[i], self.layers[i + 1])
            self.weights.append(weight_matrix)

        # Create outputs for each layer
        self.outputs = []
        for i in range(0, len(layers)):
            lst = [0] * layers[i]
            self.outputs.append(lst)

        self.interconnects = []
        for i in range(0, interconnects):
            left_i = random.randint(0, len(layers) - 1)
            left_j = random.randint(0, layers[left_i] - 1)

            right_i = random.randint(0, len(layers) - 1)
            right_j = random.randint(0, layers[right_i] - 1)

            if left_i > right_i:
                left_i, left_j, right_i, right_j = right_i, right_j, left_i, left_j

            self.interconnects.append((left_i, left_j, right_i, right_j))

        pass

    def forward(self, input_data):
        # Interconnects
        for left_i, left_j, right_i, right_j in self.interconnects:
            self.outputs[left_i][left_j] = self.outputs[right_i][right_j]

        # Calc
        self.outputs[0] = input_data

        for i in range(len(self.layers) - 1):
            self.outputs[i+1] = self.activation(np.dot(self.outputs[i], self.weights[i]))
        return self.outputs[-1]

    def calculate_error(self, target_output):
        # Mean Squared Error
        return np.mean((self.outputs[-1] - target_output) ** 2)

    def backpropagate(self, target_output, learning_rate=0.1):
        # Calculate the error
        error = self.outputs[-1] - target_output

        # Backpropagate the error
        for i in reversed(range(len(self.layers) - 1)):
            delta = error * self.activation(self.outputs[i+1], derivative=True)
            error = np.dot(delta, self.weights[i].T)
            self.weights[i] -= learning_rate * np.dot(self.outputs[i+1].T, delta)

if __name__ == '__main__':
    print()
    print('WRONG FILE :)')
