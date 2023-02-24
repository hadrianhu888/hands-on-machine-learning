import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
# Chapter 1: The Machine Learning Landscape

# load the data

data_root = "https://github.com/ageron/data/raw/main/"
lifesat = pd.read_csv(data_root + "lifesat/lifesat.csv")
X = lifesat[["GDP per capita (USD)"]].values
y = lifesat[["Life satisfaction"]].values

# display the data values

X = X.astype(np.float32)
y = y.astype(np.float32)
print(X, y)

X = X.reshape(-1, 1)
y = y.reshape(-1, 1)
print(X.shape, y.shape)

# visualize the data

lifesat.plot(kind='scatter', grid=True,
             x="GDP per capita (USD)", y="Life satisfaction")
plt.axis([20_000, 60_000, 4, 9])
plt.show()

# Select a linear model

model = LinearRegression()
model_1 = KNeighborsRegressor(n_neighbors=3)
model_2 = DecisionTreeRegressor()

# Train the model
x_train = X[:100]
y_train = y[:100]

# make a prediction for Cyprus
model.fit(X, y)
model_1.fit(X, y)
model_2.fit(X, y)

X_new = [[22587]]  # Cyprus' GDP per capita in 2020
X_new_1 = [[37_655.2]]  # Cyprus' GDP per capita in 2020
X_new_2 = [[37_655.2]]  # Cyprus' GDP per capita in 2020

print(model.predict(X_new))  # outputs [[6.30165767]]
print(model_1.predict(X_new_1))  # outputs [[6.30165767]]
print(model_2.predict(X_new_2))  # outputs [[6.30165767]]
