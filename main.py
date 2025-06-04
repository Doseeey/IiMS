import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from animate_regression import animate_regression

from get_data import get_data_energy, get_data_turbine

# MSE
def compute_cost(x, y, theta):
    m = len(y)
    predictions = x @ theta
    errors = predictions - y
    return (1 / m) * errors.T @ errors

# Gradient descent
def gradient_descent(x, y, learning_rate, n_iterations):
    m, n = x.shape
    theta = np.zeros(n)
    thetas = []
    cost_history = []

    for _ in range(n_iterations):
        y_pred = x @ theta
        error = y_pred - y
        gradient = (2 / m) * x.T @ error
        theta -= learning_rate * gradient
        
        cost_history.append(compute_cost(x, y, theta))
        thetas.append(theta.copy())

    return thetas, cost_history

# Gradient descent with momentum
def gradient_descent_with_momentum(x, y, learning_rate, n_iterations, beta):
    m, n = x.shape
    theta = np.zeros(n)
    v = np.zeros(n)
    thetas = []
    cost_history = []

    for _ in range(n_iterations):
        y_pred = x @ theta
        error = y_pred - y
        gradient = (2 / m) * x.T @ error
        v = beta * v + (1 - beta) * gradient
        theta -= learning_rate * v
        
        cost_history.append(compute_cost(x, y, theta))
        thetas.append(theta.copy())

    return thetas, cost_history

# Newton method
def newton_method(x, y, n_iterations):
    m, n = x.shape
    theta = np.zeros(n)
    thetas = []
    cost_history = []

    for _ in range(n_iterations):
        y_pred = x @ theta
        error = y_pred - y
        gradient = (2 / m) * x.T @ error
        hessian = (2 / m) * x.T @ x

        delta = np.linalg.solve(hessian, gradient)
        theta -= delta

        cost_history.append(compute_cost(x, y, theta))
        thetas.append(theta.copy())

    return thetas, cost_history

X, y = get_data_turbine()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

learning_rate = 0.01
iterations = 1000
beta = 0.9
thetas_gd, cost_history_gd = gradient_descent(X_train, y_train, learning_rate, iterations)
thetas_gdm, cost_history_gdm = gradient_descent_with_momentum(X_train, y_train, learning_rate, iterations, beta)
thetas_newton, cost_history_newton = newton_method(X_train, y_train, iterations)

def present(x, y, iters, cost_history, thetas, title, plot=True):
    if plot:
        plt.plot(range(iters), cost_history)
        plt.xlabel("Liczba iteracji")
        plt.ylabel("Funkcja kosztu (MSE)")
        plt.title(f"{title} – zbieżność")
        plt.grid(True)
        plt.show()

    theta_final = thetas[-1]
    theta_snapshots = thetas[::100]

    y_pred = x @ theta_final
    print(f"\nMSE {title}: {mean_squared_error(y, y_pred):.2f}")
    print(f"R² score {title}: {r2_score(y, y_pred):.2f}")

    animate_regression(x, y, theta_snapshots, 1, title.lower().replace(" ", "_"))

print("\n\n")
present(X_test, y_test, iterations, cost_history_gd, thetas_gd, "Gradient Descent")
present(X_test, y_test, iterations, cost_history_gdm, thetas_gdm, "Gradient Descent with Momentum")
present(X_test, y_test, iterations, cost_history_newton, thetas_newton, "Newton method")
