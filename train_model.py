# train_model.py
import numpy as np
from sklearn.linear_model import LinearRegression

# Sample dataset
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 5, 4, 5])

# Create and train a linear regression model
model = LinearRegression()
model.fit(X, y)

# Print the model's coefficients
print("Coefficients:", model.coef_)