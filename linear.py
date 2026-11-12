%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

data = fetch_california_housing()

X = data.data[:, 0].reshape(-1, 1)

y = data.target
print("Dataset loaded successfully")
print("Number of samples:", len(X))

X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.2, random_state=42
)
mean = X_train.mean()
std = X_train.std()
X_train_scaled = (X_train - mean) / std
X_test_scaled = (X_test - mean) / std

def gradient_descent(X, y, learning_rate=0.01, iterations=1000):
    n = len(y)
    w = 0.0
    b = 0.0
    for i in range(iterations):
        y_pred = w * X[:, 0] + b
        dw = (-2 / n) * np.sum(X[:, 0] * (y - y_pred))
        db = (-2 / n) * np.sum(y - y_pred)
        w = w - learning_rate * dw
        b = b - learning_rate * db
    return w, b

w_gd, b_gd = gradient_descent(X_train_scaled, y_train, learning_rate=0.01, iterations=1000
)
y_pred_gd = w_gd * X_test_scaled[:, 0] + b_gd
mse_gd = mean_squared_error(y_test, y_pred_gd)
r2_gd = r2_score(y_test, y_pred_gd)
print("\nGradient Descent Results")
print("Weight:", w_gd)
print("Bias:", b_gd)
print("MSE:", mse_gd)
print("R-squared:", r2_gd)

X_train_bias = np.c_[np.ones(X_train_scaled.shape[0]), X_train_scaled]
X_test_bias = np.c_[np.ones(X_test_scaled.shape[0]), X_test_scaled]

theta = np.linalg.pinv(X_train_bias.T @ X_train_bias) @ X_train_bias.T @ y_train
b_ne = theta[0]
w_ne = theta[1]

y_pred_ne = X_test_bias @ theta

mse_ne = mean_squared_error(y_test, y_pred_ne)
r2_ne = r2_score(y_test, y_pred_ne)
print("\nNormal Equation Results")
print("Weight:", w_ne)
print("Bias:", b_ne)
print("MSE:", mse_ne)
print("R-squared:", r2_ne)

print("\nModel Comparison")
print("-------------------------------------------")
print("Method\t\t\tMSE\t\tR-squared")
print("-------------------------------------------")
print(f"Gradient Descent\t{mse_gd:.4f}\t\t{r2_gd:.4f}")
print(f"Normal Equation\t\t{mse_ne:.4f}\t\t{r2_ne:.4f}")

sort_index = np.argsort(X_test[:, 0])
X_sorted = X_test[sort_index]
y_sorted = y_test[sort_index]
y_pred_sorted = y_pred_ne[sort_index]
plt.figure(figsize=(10, 6))
plt.scatter(X_sorted, y_sorted, alpha=0.4, label="Actual Data")
plt.plot(X_sorted, y_pred_sorted,linewidth=2,label="Fitted Regression Line")
plt.xlabel("Median Income")
plt.ylabel("Median House Value")
plt.title("Simple Linear Regression - California Housing Dataset")
plt.legend()
plt.grid(True)
plt.show()
