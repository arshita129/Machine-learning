#Implementation of gradient descent algorithm form scratch.
# Author: Arshita

# Linear regression
# y = mx + c

# Loss function: MSE (Mean Squared Error)
# MSE = (1/n) * Σ(y_i - (mx_i + c))

# Gradient descent update rules:
# m = m - learning_rate * dMSE/dm
# c = c - learning_rate * dMSE/dc

# Importing necessary libraries
import numpy as np

# Generating synthetic data for linear regression
X = np.random.rand(100, 1) * 10  # 100 random samples between 0 and 10
Y = 2.5 * X + np.random.randn(100, 1)*0.5 # Linear relationship with some noise

# Random noise means that the data points will not lie perfectly on the line defined by y = 2.5x, but will be scattered around it. 
# This makes the problem more realistic and allows us to test the effectiveness of our gradient descent algorithm in finding the best fit line.


print(X, Y)

# Initializing parameters
m = 0.0  # Slope
c = 0.0  # Interceptß
learning_rate = 0.01
epochs = 100

# Gradient descent algorithm
for epoch in range(epochs):
    # Predictions
    Y_pred = m * X + c
    
    # Calculating gradients
    # MSE = (1/n) * Σ(y_i - (mx_i + c))^2
    # dMSE/dm = (-2/n) * Σ(x_i * (y_i - (mx_i + c)))
    # dMSE/dc = (-2/n) * Σ(y_i - (mx_i + c))
    dMSE_dm = (-2 / len(X)) * np.sum(X * (Y - Y_pred))
    dMSE_dc = (-2 / len(X)) * np.sum(Y - Y_pred)
    
    # Updating parameters
    m -= learning_rate * dMSE_dm
    c -= learning_rate * dMSE_dc
    
    # Printing the loss and parameters every epoch
    mse = np.mean((Y - Y_pred) ** 2)
    print(f'Epoch {epoch}, MSE: {mse:.4f}, m: {m:.4f}, c: {c:.4f}')
