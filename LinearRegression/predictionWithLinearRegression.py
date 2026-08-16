import numpy as np
import matplotlib.pyplot as plt

# house features - first one is the size, second is the number of rooms
X = np.array([
    [1, 1],
    [2, 1],
    [3, 2]
])

# we predict how much it will cost
y = np.array([3, 5, 8])

def predict(X, w, b):
    return (X @ w) + b

def calculate_mse(y, predicted_values):
    return (np.pow(y - predicted_values, 2).sum() / len(predicted_values))

def compute_gradient(weights, bias, y, X):
    predicted_values = predict(X, weights, bias)

    weight_gradient = ((predicted_values - y).T @ X * 2) / len(X)
    bias_gradient = ((predicted_values - y).sum() * 2) / len(X)

    return [weight_gradient, bias_gradient]

new_w = np.array([0, 0])
new_b = 0
learning_rate = 0.01

for i in range(1, 500):
    weight_gradient, bias_gradient = compute_gradient(new_w, new_b, y, X)
    new_w = new_w - learning_rate * weight_gradient
    new_b = new_b - learning_rate * bias_gradient

print(new_w)
print(new_b)
print(calculate_mse(y, predict(X, new_w, new_b)))

# we give the model a new never seen data to predict how much it will cost
new_house = np.array([4, 2])
print(predict(new_house, new_w, new_b))
