import numpy as np

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
        self._last_input = None  # Store input for gradient calculation
        self._last_output = None  # Store output before activation

    @property
    def state_dict(self):
        return dict(
            weights=self.weights,
            bias=self.bias,
        )

    def forward(self, x, track_fn):
        """ Linear transformation + tracking """
        self._last_input = x  # Store input for backprop
        self._last_output = np.dot(x, self.weights) + self.bias  # Linear output
        track_fn.append("linear")  # Track function application in global list
        return self._last_output

    def backward(self):
        """ this should update the weights and bias """

    def __call__(self, track_fn, x=None):
        if x is None:
            return self.state_dict
        else:
            return self.forward(x, track_fn)

    def __repr__(self):
        return str(self.state_dict)


def relu_activation(x, track_fn):
    """ Applies ReLU activation function and tracks it """
    track_fn.append("relu")  # Track ReLU function for backprop
    return np.maximum(0, x)


class NeuralNetwork:
    def __init__(self):
        # Store forward pass function tracking
        self._grad_fn = []

    @property
    def state_dict(self):
        return self.__dict__

    def _get_layers(self):
        return [self.state_dict[key] for key in self.state_dict.keys() if not key.startswith('_')]

    def forward(self, x):
        raise NotImplementedError('The forward() method must be implemented')

    def __call__(self, x):
        return self.forward(x)


class CustomNeuralNetwork(NeuralNetwork):
    def __init__(self):
        super().__init__()
        self.input_layer = LinearLayer(2, 4)
        self.hidden_layer = LinearLayer(4, 4)
        self.output_layer = LinearLayer(4, 2)

    def forward(self, x):
        x = relu_activation(self.input_layer(self._grad_fn, x), self._grad_fn)
        x = relu_activation(self.hidden_layer(self._grad_fn, x), self._grad_fn)
        x = self.output_layer(self._grad_fn, x)
        return x

    def backward(self, x: LinearLayer, y_true):
        """ the method should calculate gradients and update the weights of each layer"""

        # compute loss gradient
        y_pred = self.forward(x)
        dl_dY = 2 * (y_pred - y_true) / y_true.size

        # backpropagete loss to output layer linear

        # backp loss to relu of output layer
        # backpropagete loss to hidden layer linear
        # backpropagation loss to relu of hidden layer
        # backpropaget loss to input layer linear

    @property
    def track_functions(self) -> list:
        return self._grad_fn

    def __repr__(self):
        return self.state_dict