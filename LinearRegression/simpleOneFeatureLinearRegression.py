import numpy as np

# our x values - the feature values
X = np.array([1, 2, 3, 4, 5])

# our actual values - the labels
y = np.array([3, 5, 7, 9, 11])

# how do we predict the label value with linear regression?
# y = b + w*x
# a numpy az array minden elemére elvégzi a műveletet
def predict(X, w, b):
    return X * w + b

# mse = Mean Squared Error
# the sum of the actual minus the predicted values, squared, divided by the number of elements in set
def calculate_mse(y, predicted_values):
    return (np.pow(y - predicted_values, 2).sum() / len(predicted_values))

# we have to see if we add to the w if the mse is getting greater or not
# if it the mse is smaller then we can say that the gradient is negative - imagine the gradient as the slope of the function
# if we go right - with w - we go down - with mse - so the slope is going downwards
# if we add to w and the mse gets greater than the gradient is posive and its going upwards

# we need to compute the gradient multiple times because the gradient is only valid for a given point
# it gives the size, magnitude of the slope and the direction of it from the given point
def compute_gradient(weight, bias, y, X, learning_rate):
    predicted_values = predict(X, weight, bias)
    new_weight = weight - (learning_rate * (((predicted_values - y) * 2 * X).sum() / len(X)))
    new_bias = bias - (learning_rate * 2 * (((predicted_values - y).sum()) / len(X)))
    print("The new weight is: ", new_weight)
    print("The new bias is: ", new_bias)
    return [new_weight, new_bias]

new_w = 1
new_b = 0
for i in range(1, 10):
    [new_w, new_b] = compute_gradient(new_w, new_b, y, X, 0.01)
    print("The loss is: ", calculate_mse(y, predict(X, new_w, new_b)))
