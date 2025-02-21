import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import SGD, Adam

import matplotlib.pyplot as plt


def print_nn_state(network):
    for key in network.state_dict.keys():
        if not key.startswith('_'):
            layer = network.state_dict[key]
            print(f'{key} input: ', layer._last_input)
            print(f'{key} weights: \n', layer.state_dict['weights'])
            print(f'{key} bias: ', layer.state_dict['bias'])
            print(f'{key} output: ', layer._last_output)
            print('=' * 60)


class Bias:
    def __init__(self, h_size: int) -> None:
        self._h_size = h_size
        self._value = None

    @property
    def value(self):
        if self._value is None:
            self._value = self._generate_value()

        return self._value

    def _generate_value(self):
        return np.random.uniform(-0.5, 0.5, size=(1, self._h_size))

    def __call__(self):
        return self.value

    def __repr__(self):
        return str(self.__dict__)


class Weight:
    def __init__(self, v_size: int, h_size: int):
        self._v_size = v_size
        self._h_size = h_size
        self._value = None

    @property
    def value(self):
        if self._value is None:
            self._value = self._generate_value()

        return self._value

    def _generate_value(self):
        return np.random.uniform(-0.5, 0.5, size=(self._v_size, self._h_size))

    def __call__(self):
        return self.value

    def __repr__(self):
        return str(self.__dict__)


class LinearLayer:
    def __init__(self, in_features: int, out_features: int, require_grad=True):
        self.in_features = in_features
        self.out_features = out_features
        self.bias = Bias(out_features)()
        self.weights = Weight(in_features, out_features)()
        self._require_grad = require_grad
        self._input = None  # store input values either activated or not
        self._z_output = None  # Store output before activation
        self._a_output = None  # Store output after activation

    @property
    def state_dict(self):
        return dict(
            weights=self.weights,
            bias=self.bias,
        )

    def forward(self, x):
        """ Linear transformation W.X + B """
        self._input = x  # save the input
        self._z_output = np.dot(self._input, self.weights) + self.bias  # save values before activation
        return self._z_output

    def backward(self, dldy, lr):
        """ Backpropagate gradients and update weights & bias """

        # Compute gradient of weights and activations
        if self._a_output is None:  # No activation applied (e.g., output layer)
            dw = np.dot(self._input.T, dldy)  # Gradient of weights
            da = np.dot(dldy, self.weights.T)  # Gradient of activations for previous layer
        else:  # ReLU activation applied
            dZ = dldy * relu_derivative(self._z_output)  # Apply ReLU derivative
            dw = np.dot(self._input.T, dZ)  # Compute weight gradients
            da = np.dot(dZ, self.weights.T)  # Compute activation gradients for previous layer

        # Compute gradient of bias (sum over batch)
        db = np.sum(dldy, axis=0, keepdims=True)

        # Update weights and biases using gradient descent
        self.weights -= lr * dw  # Correct weight update
        self.bias -= lr * db  # Correct bias update

        return da  # Return gradient for previous layer

    def __call__(self, x):
        if x is None:
            return self.state_dict
        else:
            return self.forward(x)

    def __repr__(self):
        return str(self.state_dict)


class NeuralNetwork:

    @property
    def state_dict(self):
        return self.__dict__

    def _get_layers(self):
        return [self.state_dict[key] for key in self.state_dict.keys() if not key.startswith('_')]

    def forward(self, x):
        raise NotImplementedError('The forward() method must be implemented')

    def backward(self, y_pred, y_true, lr=0.01):
        dldy = 2 * (y_pred - y_true) / y_true.size

        for key in reversed(self.state_dict.keys()):
            layer = self.__getattribute__(key)
            dldy = layer.backward(dldy, lr=lr)

    def __call__(self, x):
        return self.forward(x)

    @property
    def track_functions(self) -> list:
        return self._grad_fn

    def __repr__(self):
        return self.state_dict


def relu_activation(layer, x):
    """ Applies ReLU activation function and tracks it """
    layer._a_output = np.maximum(0, x)
    return layer._a_output


def relu_derivative(x):
    return (x > 0).astype(float)


class TestNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layer = nn.Linear(2, 2)
        self.hidden_layer = nn.Linear(2, 4)
        self.output_layer = nn.Linear(4, 3)

    def forward(self, x):
        x = torch.tensor(x, dtype=torch.float32)
        x = F.relu(self.input_layer(x))
        x = F.relu(self.hidden_layer(x))
        return F.relu(self.output_layer(x))


def train_original(model, optimizer, epochs, x_true, y_true):
    losses = []
    y_true = torch.tensor(y_true, dtype=torch.float32)
    for epoch in range(epochs):
        y_pred = model.forward(x_true)
        loss = F.mse_loss(y_pred, y_true)
        losses.append(loss.detach().numpy())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0:  # Print loss every 100 epochs
            print(f"Epoch {epoch}, Loss: {loss:.4f}")
    return losses


class SingleNeuronNN(NeuralNetwork):
    def __init__(self):
        super().__init__()
        self.input_layer = LinearLayer(2, 2)
        self.hidden_layer = LinearLayer(2, 4)
        self.output_layer = LinearLayer(4, 3)

    def forward(self, x):
        x = self.input_layer(x)
        x = relu_activation(self.input_layer, x)
        x = self.hidden_layer(x)
        x = relu_activation(self.hidden_layer, x)
        x = self.output_layer(x)
        return relu_activation(self.output_layer, x)


def train(model, x, y_pred, y_true, epochs=1, lr=0.05):
    losses = []
    for epoch in range(epochs):
        y_pred = model.forward(x)

        # loss MSE
        loss = np.mean((y_pred - y_true) ** 2) / y_true.size
        losses.append(loss)
        print(f'Epoch: {epoch}, Loss: ', loss)
        model.backward(y_pred, y_true, lr=lr)
    return losses


if __name__ == '__main__':
    x_true = np.random.randn(1, 2)
    y_true = np.random.randn(1, 3)
    y_pred = np.random.randn(1, 3)

    single_neuron = SingleNeuronNN()
    test_network = TestNetwork()
    optimizer = Adam(test_network.parameters(), lr=0.03)

    losses_original = train_original(test_network, optimizer, epochs=200, x_true=x_true, y_true=y_true)
    losses = train(single_neuron, x=x_true, y_pred=y_pred, y_true=y_true, epochs=200, lr=0.03)

    plt.plot(range(len(losses)), losses)
    plt.plot(range(len(losses_original)), losses_original)
    plt.ylabel('Loss')
    plt.xlabel('Iteration')
    plt.show()
