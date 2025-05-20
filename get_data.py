import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import StandardScaler

def get_data_energy():
    # https://archive.ics.uci.edu/dataset/242/energy+efficiency
    energy_efficiency = fetch_ucirepo(id=242)

    X = energy_efficiency.data.features

    # Dataset has 2 target values:
    #  - y1 - Heating Efficiency
    #  - y2 - Cooling Efficiency
    # We can focus on Heating efficiency only or if we want
    # to make 3d then on both
    y = energy_efficiency.data.targets.values[:, 1]

    _, features = X.shape

    # for i in range(features):
    #     print(X.values[:, i].min(), X.values[:, i].max()) 

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X = np.c_[np.ones((X.shape[0], 1)), X] #bias

    # for i in range(features):
    #     print(X[:, i].min(), X[:, i].max()) 
    return X, y

def get_data_turbine():
    # https://archive.ics.uci.edu/dataset/994/micro+gas+turbine+electrical+energy+prediction
    df = pd.read_csv("data/gas_turbine_energy_dataset.csv")

    X = df["input_voltage"].to_numpy().reshape(-1, 1)  # shape: (n_samples, 1)
    y = df["el_power"].to_numpy()  

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X = np.c_[np.ones((X.shape[0], 1)), X] #bias

    return X, y