# Import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import numpy as np

# Generate some random data
np.random.seed(0)
X = np.random.rand(100, 1)
y = 3 + 2 * X + np.random.randn(100, 1) / 1.5

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("Coefficient of determination (R^2):", metrics.r2_score(y_test, y_pred))
print("Mean squared error:", metrics.mean_squared_error(y_test, y_pred))
print("Root mean squared error:", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

# Print coefficients
print("Intercept:", model.intercept_)
print("Slope:", model.coef_)
