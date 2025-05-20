import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from animate_regression import animate_regression

from get_data import get_data_energy, get_data_turbine

# Dummy z GPT
def compute_cost(X, y, theta):
    m = len(y)
    predictions = X.dot(theta)
    errors = predictions - y
    return (1 / (2 * m)) * np.dot(errors.T, errors)

# Gradient descent
def gradient_descent(X, y, theta, learning_rate, n_iterations):
    m = len(y)
    cost_history = []

    for i in range(n_iterations):
        gradients = (1 / m) * X.T.dot(X.dot(theta) - y)
        theta -= learning_rate * gradients
        cost = compute_cost(X, y, theta)
        cost_history.append(cost)   
    return theta, cost_history

def gradient_descent_with_snapshots(X, y, theta, learning_rate, n_iterations, snapshot_every=100):
    m = len(y)
    theta_snapshots = []
    cost_history = []

    for i in range(n_iterations):
        gradients = (1 / m) * X.T @ (X @ theta - y)
        theta -= learning_rate * gradients

        if i % snapshot_every == 0:
            theta_snapshots.append(theta.copy())
            
        cost_history.append(compute_cost(X, y, theta))

    return theta, theta_snapshots, cost_history

X, y = get_data_turbine()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

theta = np.zeros(X.shape[1])
learning_rate = 0.01
iterations = 1000

theta_final, theta_snapshots, cost_history = gradient_descent_with_snapshots(X, y, theta, learning_rate, iterations)

y_train_pred = X_train @ theta_final
y_test_pred = X_test @ theta_final

plt.plot(range(iterations), cost_history)
plt.xlabel("Liczba iteracji")
plt.ylabel("Funkcja kosztu (MSE)")
plt.title("Gradient Descent – zbieżność")
plt.grid(True)
plt.show()


y_pred = X.dot(theta_final)
print(f"MSE: {mean_squared_error(y, y_pred):.2f}")
print(f"R² score: {r2_score(y, y_pred):.2f}")

animate_regression(X, y, theta_snapshots, 1)



