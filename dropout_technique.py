# import numpy to perform matrix operations and random number generations.
import numpy as np


# Defined a class called NeuralNetwork to encapsulate all functionalities of the neural network.
class NeuralNetwork:
    # Constructor of the NeuralNetwork class. It initializes the neural network with given layer sizes.
    def __init__(self, layer_sizes):
        # Initialize weights for each layer. The weights between two successive layers of size m and n have the shape (m, n).
        self.weights = [
            np.random.randn(m, n) for m, n in zip(layer_sizes[:-1], layer_sizes[1:])
        ]
        # Biases are initialized for each neuron except the input layer.
        self.biases = [np.random.randn(1, n) for n in layer_sizes[1:]]
        # Store the provided layer sizes.
        self.layer_sizes = layer_sizes

    # Defines the sigmoid activation function, which squashes the input between 0 and 1.
    # def sigmoid(self, z):
    #     return 1 / (1 + np.exp(-z))
    # Defines the tanh activation function, which squashes the input between 0 and 1.
    def tanh(self, z):
        return np.tanh(z)

    # Derivative of the sigmoid function, used in backpropagation to compute gradients.
    # def sigmoid_prime(self, z):
    #     return self.sigmoid(z) * (1 - self.sigmoid(z))
    # Derivative of the tanh function, used in backpropagation to compute gradients.
    def tanh_prime(self, z):
        return 1 - np.tanh(z) ** 2

    # This is the forward propagation method. Given input a, it computes the output for each layer. If dropout masks are provided, they're applied to the activations.
    def forward(self, a, dropout_masks=None):
        # We store activations for all layers, starting with the input.
        activations = [a]
        # Loop over each layer's weights and biases.
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            # Compute the weighted sum for each neuron.
            z = np.dot(a, w) + b
            # Apply the sigmoid activation function.
            # a = self.sigmoid(z)
            # Apply the tanh activation function.
            a = self.tanh(z)
            # If dropout masks are provided, apply them to the activations (but not to the output layer).
            if (
                dropout_masks and i < len(self.weights) - 1
            ):  # Apply dropout, except for the output layer
                a *= dropout_masks[i]
            # Store the activations for this layer.
            activations.append(a)
        # Return the activations for all layers.
        return activations

    # This is the backpropagation method. Given input x and target output y, it computes the gradient of the loss function w.r.t. every weight and bias in the network.
    def backpropagation(self, x, y, dropout_masks=None):
        # Perform forward propagation to get activations.
        activations = self.forward(x, dropout_masks)
        # Number of training samples.
        m = x.shape[0]
        # Compute the error at the output layer.
        n_layers = len(self.layer_sizes)

        # Output error
        delta = activations[-1] - y

        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]

        nabla_w[-1] = np.dot(activations[-2].T, delta) / m
        nabla_b[-1] = np.sum(delta, axis=0, keepdims=True) / m

        # Backpropagate error
        for l in range(n_layers - 3, -1, -1):
            z = np.dot(activations[l], self.weights[l])
            # sp = self.sigmoid_prime(z)
            sp = self.tanh_prime(z)
            delta = np.dot(delta, self.weights[l + 1].T) * sp
            if dropout_masks:
                delta *= dropout_masks[l]
            nabla_w[l] = np.dot(activations[l].T, delta) / m
            nabla_b[l] = np.sum(delta, axis=0, keepdims=True) / m

        return (nabla_w, nabla_b)

    # The training method. It updates weights and biases using gradient descent and the computed gradients.
    def train(self, data, labels, epochs, learning_rate, dropout_rate=0.5):
        # Repeat the training process for a specified number of epochs.
        for epoch in range(epochs):
            # Generate dropout masks for the hidden layers.
            dropout_masks = [
                np.random.binomial(1, 1 - dropout_rate, size=(data.shape[0], size))
                for size in self.layer_sizes[1:-1]
            ]
            # Compute the gradients for weights and biases using backpropagation.
            nabla_w, nabla_b = self.backpropagation(data, labels, dropout_masks)

            # Update weights and biases using gradient descent.
            self.weights = [
                w - learning_rate * dw for w, dw in zip(self.weights, nabla_w)
            ]
            self.biases = [
                b - learning_rate * db for b, db in zip(self.biases, nabla_b)
            ]

            # Get the network's predictions for the current weights and biases.
            predictions = self.forward(data)[-1]
            # Compute the cross-entropy loss.
            loss = -np.mean(
                labels * np.log(predictions) + (1 - labels) * np.log(1 - predictions)
            )
            # Compute the accuracy.
            accuracy = np.mean((np.round(predictions) == labels).all(axis=1))
            # Print the loss and accuracy for each epoch.
            print(f"Epoch {epoch+1}: Loss: {loss}, Accuracy: {accuracy}")


# Dummy data
data = np.array([[0, 0, 0, 0], [1, 1, 1, 1], [0, 1, 0, 1], [1, 0, 1, 0]])
labels = np.array([[1, 0], [0, 1], [0, 1], [1, 0]])

# Initialize and train
nn = NeuralNetwork([4, 6, 6, 2])
nn.train(data, labels, epochs=500, learning_rate=0.1)
