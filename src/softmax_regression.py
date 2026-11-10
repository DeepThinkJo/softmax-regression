import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("../data/Iris.csv")

# Preprocessing
x = df.drop(columns=["Id", "Species"]).to_numpy()
y = df["Species"].to_numpy()

# One-hot encoding
classes, indicies = np.unique(y, return_inverse=True)
Y = np.eye(len(classes), dtype=int)[indicies]

# Feature Scaling: Standardization
mean = np.mean(x, axis=0)
std = np.std(x, axis=0)
x = (x - mean) / std

# Design matrix
X = np.hstack((np.ones((x.shape[0], 1)), x))

m = X.shape[0]  # Number of samples
n = X.shape[1]  # Number of features (including bias term)
K = Y.shape[1]  # Number of classes

# Softmax Regression Model
param = np.zeros((n, K))

epochs = 10000
lr = 10 ** (-2)

loss_history = []

def softmax(z):
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

for epoch in range(epochs):
    Z = X @ param
    pred = softmax(Z)
    param -= lr * (1.0 / m) * X.T @ (pred - Y)
    loss = np.mean(-np.sum(Y * np.log(pred), axis=1))
    loss_history.append({"iteration": epoch, "loss": loss})

loss_history_df = pd.DataFrame(loss_history)
loss_history_df.to_csv("../results/csv/loss_history.csv", index=False)