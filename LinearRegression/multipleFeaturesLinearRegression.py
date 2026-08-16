import numpy as np
import matplotlib.pyplot as plt

X = np.array([
    [1, 1],
    [2, 1],
    [3, 2]
])

y = np.array([3, 5, 8])

# @ is the matrix multiplication
def predict(X, w, b):
    return (X @ w) + b

def calculate_mse(y, predicted_values):
    return (np.pow(y - predicted_values, 2).sum() / len(predicted_values))

# note that X.T is the transpose of the matrix
# it is needed to be transposed because 3x2 * 3x1 cannot be done, but 2x3 * 3*1 yeah because the inside dimensions are the same
# the transposed matrix needs to be the first one because we need a vector which is 1x2 dimensions for the weight vector
# in the other case the multipication couldnt be done because of the dimensions
# the other way it can be done if we would write this: (predicted_values - y).T @ X
# in this case there isnt any sum because we would be summing up the 2x1 vector's numbers which is not what we want
# the sum is "hidden" in the matrix multiplication
def compute_gradient(weights, bias, y, X, learning_rate):
    predicted_values = predict(X, weights, bias)
    new_weights = weights - (learning_rate * (( (predicted_values - y).T @ X * 2) / len(X)))
    new_bias = bias - (learning_rate * 2 * (((predicted_values - y).sum()) / len(X)))
    print("The new weight is: ", new_weights[0], new_weights[1])
    print("The new bias is: ", new_bias)
    return [new_weights, new_bias]

new_w = np.array([0, 0])
new_b = 0
for i in range(1, 500):
    [new_w, new_b] = compute_gradient(new_w, new_b, y, X, 0.01)
    print("The loss is: ", calculate_mse(y, predict(X, new_w, new_b)))
