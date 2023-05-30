# Setup

import sys

assert sys.version_info >= (3, 5)

from packaging import version
import sklearn
from sklearn import __version__ as sklearn_version

assert version.parse(sklearn.__version__) >= version.parse("0.20")

import matplotlib.pyplot as plt

plt.rc("font", family="serif", size=12)
plt.rc("axes", labelsize=12)
plt.rc("xtick", labelsize=12)
plt.rc("ytick", labelsize=12)
plt.rc("legend", fontsize=12)
plt.rc("figure", titlesize=12)

from pathlib import Path as path

IMAGES_PATH = path("images")
IMAGES_PATH.mkdir(parents=True, exist_ok=True)


def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = IMAGES_PATH / fig_id
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


# Linear Regression

import numpy as np

np.random.seed(42)
m = 100
X = 2 * np.random.rand(m, 1)
y = 4 + 3 * X + np.random.randn(m, 1)

import matplotlib.pyplot as plt

plt.figure(figsize=(9, 9))
plt.plot(X, y, "b.")
plt.xlabel("$x_1$", fontsize=18)
plt.ylabel("$y$", rotation=0, fontsize=18)
plt.axis([0, 2, 0, 15])
plt.title("Generated Data")
plt.show()
save_fig("generated_data_plot")

from sklearn.preprocessing import add_dummy_feature

X_b = add_dummy_feature(X)
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

# Linear Regression using the Normal Equation

print(theta_best)

X_new = np.array([[0], [2]])
X_new_b = add_dummy_feature(X_new)
y_predict = X_new_b.dot(theta_best)
print(y_predict)

import matplotlib.pyplot as plt

plt.figure(figsize=(9, 9))
plt.plot(X_new, y_predict, "r-", linewidth=2, label="Predictions")
plt.plot(X, y, "b.")

plt.xlabel("$x_1$", fontsize=18)
plt.ylabel("$y$", rotation=0, fontsize=18)
plt.axis([0, 2, 0, 15])
plt.title("Linear Regression Predictions")
plt.legend(loc="upper left", fontsize=14)
plt.show()

save_fig("linear_regression_predictions_plot")

from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(X, y)
print(lin_reg.intercept_, lin_reg.coef_)

print(lin_reg.predict(X_new))

theta_best_svd, residuals, rank, s = np.linalg.lstsq(X_b, y, rcond=1e-6)
print(theta_best_svd)

print(np.linalg.pinv(X_b).dot(y))

# Linear Regression using Batch Gradient Descent

np.linalg.pinv(X_b) @ y

eta = 0.1

# Gradient Descent

eta = 0.1
n_epochs = 1000
m = len(X_b)

np.random.seed(42)
theta = np.random.randn(2, 1)

for epoch in range(n_epochs):
    gradients = 2 / m * X_b.T.dot(X_b.dot(theta) - y)
    theta = theta - eta * gradients

print(theta)

X_new_b.dot(theta)

theta_path_bgd = []


def plot_gradient_descent(theta, eta):
    m = len(X_b)
    plt.figure(figsize=(9, 9))
    plt.plot(X, y, "b.")
    n_epochs = 1000
    theta_path = []
    for epoch in range(n_epochs):
        if epoch < 10:
            y_predict = X_new_b.dot(theta)
            style = "b-" if epoch > 0 else "r--"
            plt.plot(X_new, y_predict, style)
        gradients = 2 / m * X_b.T.dot(X_b.dot(theta) - y)
        theta = theta - eta * gradients
        theta_path.append(theta)
    plt.xlabel("$x_1$", fontsize=18)
    plt.axis([0, 2, 0, 15])
    plt.grid()
    plt.title(r"$\eta = {}$".format(eta), fontsize=16)


print(theta)

np.random.seed(42)
theta = np.random.randn(2, 1)

plt.figure(figsize=(10, 4))
plt.subplot(131)
plot_gradient_descent(theta, eta=0.02)
plt.ylabel("$y$", rotation=0)
plt.subplot(132)
theta_path_bgd = plot_gradient_descent(theta, eta=0.1)
plt.gca().axes.yaxis.set_ticklabels([])
plt.subplot(133)
plt.gca().axes.yaxis.set_ticklabels([])
plot_gradient_descent(theta, eta=0.5)
save_fig("gradient_descent_plot")
plt.show()

# Stochastic Gradient Descent

theta_path_sgd = []

n_epochs = 50
t0, t1 = 5, 50


def learning_schedule():
    return t0 / (t1 + epoch)


np.random.seed(42)
theta = np.random.randn(2, 1)

n_shown = 20
plt.figure(figsize=(9, 9))

for epoch in range(n_epochs):
    for iterations in range(m):
        if epoch == 0 and iterations < n_shown:
            y_predict = X_new_b.dot(theta)
            style = "b-" if iterations > 0 else "r--"
            plt.plot(X_new, y_predict, style)
        random_index = np.random.randint(m)
        xi = X_b[random_index : random_index + 1]
        yi = y[random_index : random_index + 1]
        gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
        eta = learning_schedule()
        theta = theta - eta * gradients
        theta_path_sgd.append(theta)

plt.plot(X, y, "b.")
plt.xlabel("$x_1$", fontsize=18)
plt.ylabel("$y$", rotation=0, fontsize=18)
plt.axis([0, 2, 0, 15])
plt.title("Stochastic Gradient Descent")
plt.show()
save_fig(fig_path="sgd_plot")

print(theta)

from sklearn.linear_model import SGDRegressor

sgd_reg = SGDRegressor(
    max_iter=50, tol=-np.infty, penalty=None, eta0=0.1, random_state=42
)
sgd_reg.fit(X, y.ravel())

print(sgd_reg.intercept_, sgd_reg.coef_)

# Mini-batch Gradient Descent

from math import ceil as ceil

n_epochs = 50
minibatch_size = 20
n_batches_per_epoch = ceil(m / minibatch_size)

np.random.seed(42)
theta = np.random.randn(2, 1)

t0, t1 = 200, 1000


def learning_schedule(t):
    return t0 / (t + t1)


theta_path_mgd = []
for epoch in range(n_epochs):
    for batch_index in range(n_batches_per_epoch):
        if epoch == 0 and batch_index < 10:
            y_predict = X_new_b.dot(theta)
            style = "b-" if batch_index > 0 else "r--"
            plt.plot(X_new, y_predict, style)
        random_index = np.random.randint(m)
        xi = X_b[random_index : random_index + minibatch_size]
        yi = y[random_index : random_index + minibatch_size]
        gradients = 2 / minibatch_size * xi.T.dot(xi.dot(theta) - yi)
        eta = learning_schedule(epoch * n_batches_per_epoch + batch_index)
        theta = theta - eta * gradients
        theta_path_mgd.append(theta)

theta_path_bdg = np.array(theta_path_bgd)
theta_path_sgd = np.array(theta_path_sgd)
theta_path_mgd = np.array(theta_path_mgd)

plt.figure(figsize=(9, 9))
plt.plot(
    theta_path_sgd[:, 0], theta_path_sgd[:, 1], "r-s", linewidth=1, label="Stochastic"
)
plt.plot(
    theta_path_mgd[:, 0], theta_path_mgd[:, 1], "g-+", linewidth=2, label="Mini-batch"
)
plt.plot(theta_path_bdg[:, 0], theta_path_bdg[:, 1], "b-o", linewidth=3, label="Batch")
plt.legend(loc="upper left", fontsize=16)
plt.xlabel(r"$\theta_0$", fontsize=20)
plt.ylabel(r"$\theta_1$", fontsize=20, rotation=0)
plt.grid()
plt.axis([2.5, 4.5, 2.3, 3.9])
save_fig("gradient_descent_paths_plot")
plt.show()

# Polynomial Regresssion

np.random.seed(42)
m = 100
X = 6 * np.random.rand(m, 1) - 3
y = 0.5 * X**2 + X + 2 + np.random.randn(m, 1)

plt.figure(figsize=(9, 9))
plt.plot(X, y, "b.")
plt.xlabel("$x_1$", fontsize=18)
plt.ylabel("$y$", rotation=0, fontsize=18)
plt.axis([-3, 3, 0, 10])
save_fig("quadratic_data_plot")
plt.show()

from sklearn.preprocessing import PolynomialFeatures

poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)
print(X[0])

print(X_poly[0])

lin_reg = LinearRegression()
lin_reg.fit(X_poly, y)
print(lin_reg.intercept_, lin_reg.coef_)
X_new = np.linspace(-3, 3, 100).reshape(100, 1)

X_new_poly = poly_features.transform(X_new)
y_new = lin_reg.predict(X_new_poly)

plt.figure(figsize=(9, 9))
plt.plot(X, y, "b.")
plt.plot(X_new, y_new, "r-", linewidth=2, label="Predictions")
plt.xlabel("$x_1$", fontsize=18)
plt.ylabel("$y$", rotation=0, fontsize=18)
plt.legend(loc="upper left", fontsize=14)
plt.axis([-3, 3, 0, 10])
save_fig("quadratic_predictions_plot")
plt.show()

poly_features = PolynomialFeatures(degree=10, include_bias=False)
X_poly = poly_features.fit_transform(X)
lin_reg = LinearRegression()
lin_reg.fit(X_poly, y)
y_newbig = lin_reg.predict(X_new_poly)

plt.figure(figsize=(9, 9))
plt.plot(X, y, "b.")
plt.plot(X_new, y_newbig, "r-", linewidth=2, label="Predictions")
plt.xlabel("$x_1$", fontsize=18)
plt.ylabel("$y$", rotation=0, fontsize=18)
plt.legend(loc="upper left", fontsize=14)
plt.axis([-3, 3, 0, 10])
save_fig("high_degree_polynomials_plot")
plt.show()

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

plt.figure(figsize=(9, 9))

for style, width, dgree in (("g-", 1, 300), ("b--", 2, 2), ("r-+", 2, 1)):
    polybig_features = PolynomialFeatures(degree=dgree, include_bias=False)
    std_scaler = StandardScaler()
    lin_reg = LinearRegression()
    polynomial_regression = Pipeline(
        [
            ("poly_features", polybig_features),
            ("std_scaler", std_scaler),
            ("lin_reg", lin_reg),
        ]
    )
    polynomial_regression.fit(X, y)
    y_newbig = polynomial_regression.predict(X_new)
    plt.plot(
        X_new,
        y_newbig,
        style,
        label=str(dgree),
        linewidth=width,
    )

plt.plot(X, y, "b.", linewidth=3)
plt.legend(loc="upper left")
plt.xlabel("$x_1$", fontsize=18)
plt.ylabel("$y$", rotation=0, fontsize=18)
plt.axis([-3, 3, 0, 10])
save_fig("high_degree_polynomials_plot")
plt.show()

# Learning Curves

from sklearn.model_selection import learning_curve

train_sizes, train_scores, test_scores = learning_curve(
    LinearRegression(),
    X,
    y,
    cv=10,
    scoring="neg_mean_squared_error",
    train_sizes=np.linspace(0.1, 1, 10),
    scoring="neg_mean_squared_error",
)
train_errors = -train_scores.mean(axis=1)
test_errors = -valid_scores.mean(axis=1)

plt.figure(figsize=(9, 9))
plt.plot(
    train_sizes,
    train_errors,
    "r-+",
    linewidth=2,
    label="Training set",
)

plt.plot(
    train_sizes,
    test_errors,
    "b-",
    linewidth=3,
    label="Validation set",
)

plt.xlabel("Training set size", fontsize=14)
plt.ylabel("RMSE", fontsize=14)
plt.legend(loc="upper right", fontsize=14)

plt.show()

# Regularized Linear Models

# Ridge Regression

np.random.seed(42)
m = 20
X = 3 * np.random.rand(m, 1)
y = 1 + 0.5 * X + np.random.randn(m, 1) / 1.5

X_new = np.linspace(0, 3, 100).reshape(100, 1)

# extra code to make the plot look pretty

plt.figure(figsize=(9, 9))
plt.plot(X, y, "b.")
plt.xlabel("$x_1$", fontsize=18)
plt.ylabel("$y$", rotation=0, fontsize=18)
plt.axis([0, 3, 0, 4])
save_fig("generated_data_plot")
plt.grid()
plt.show()

from sklearn.linear_model import Ridge

ridge_reg = Ridge(alpha=1, solver="cholesky", random_state=42)


def plot_model(model_class, polynomial, alphas, **model_kwargs):
    plt.plot(X, y, "b.", linewidth=3)
    for alpha, style in zip(alphas, ("b-", "g--", "r:")):
        model = model_class(alpha, **model_kwargs) if alpha > 0 else LinearRegression()
        if polynomial:
            model = Pipeline(
                [
                    (
                        "poly_features",
                        PolynomialFeatures(degree=10, include_bias=False),
                    ),
                    ("std_scaler", StandardScaler()),
                    ("regul_reg", model),
                ]
            )
        model.fit(X, y)
        y_new_regul = model.predict(X_new)
        lw = 2 if alpha > 0 else 1
        plt.plot(
            X_new,
            y_new_regul,
            style,
            linewidth=lw,
            label=r"$\alpha = {}$".format(alpha),
        )
    plt.plot(X, y, "b.", linewidth=3)
    plt.legend(loc="upper left", fontsize=15)
    plt.xlabel("$x_1$", fontsize=18)
    plt.axis([0, 3, 0, 4])


sdg_reg = SGDRegressor(penalty="l2", random_state=42)
sdg_reg.fit(X, y.ravel())
sdg_reg.predict([[1.5]])

ridge_reg = Ridge(alpha=1, solver="cholesky", random_state=42)
ridge_reg.fit(X, y)
ridge_reg.predict([[1.5]])

alpha = 1
A = np.array([[1, alpha], [alpha, 1]])
X_b = np.c_[np.ones((2, 1)), X]  # add x0 = 1 to each instance
np.linalg.inv(X_b.T.dot(X_b) + A).dot(X_b.T).dot(y)

ridge_reg.intercept_, ridge_reg.coef_

sgd_reg = SGDRegressor(penalty="l2", random_state=42)
sdg_reg.fit(X, y)
sdg_reg.intercept_, sdg_reg.coef_

ridge_reg = Ridge(alpha=1, solver="sag", random_state=42)
ridge_reg.fit(X, y)
ridge_reg.predict([[1.5]])

sgd_reg = SGDRegressor(penalty="l2", random_state=42)
sdg_reg.fit(X, y)
sdg_reg.predict([[1.5]])

# Lasso Regression

from sklearn.linear_model import Lasso

plt.figure(figsize=(9, 9))
plot_model(Lasso, polynomial=False, alphas=(0, 0.1, 1), random_state=42)
plt.ylabel("$y$", rotation=0, fontsize=18)
save_fig("lasso_regression_plot")
plt.show()

# extra code to reproduce the book figure

plt.figure(figsize=(9, 9))
plot_model(Lasso, polynomial=True, alphas=(0, 10**-7, 1), tol=1, random_state=42)
plt.ylabel("$y$", rotation=0, fontsize=18)
plt.legend(loc="upper left", fontsize=15)
save_fig("lasso_regression_plot")
plt.show()

# extra code – this BIG cell generates and saves Figure 4–19

t1a, t1b, t2a, t2b = -1, 3, -1.5, 1.8
t1a = np.linspace(t1a, t1b, 500)
t2a = np.linspace(t2a, t2b, 500)
T = np.meshgrid(t1a, t2a)
Xr = np.c_[T[0].ravel(), T[1].ravel()]
yr = 2 + 0.5 * T[0].ravel() + 0.5 * T[1].ravel() + np.random.randn(42)

J = (Xr.T[1] < 0).astype(np.int32) + (Xr.T[0] > 2).astype(np.int32) + 2

N1 = np.linalg.norm(Xr - np.array([1, 0]).reshape(-1, 1), axis=0, ord=1)
N2 = np.linalg.norm(Xr - np.array([1, 0]).reshape(-1, 1), axis=0, ord=2)

t_min_idx = np.argmin(N1)
t_max_idx = np.argmax(N1)


def bdg_path(theta, X, y, l1, l2, core=1, n_iterations=200):
    path = [theta]
    for iterations in range(n_iterations):
        gradients = (
            2 / m * X_b.T.dot(X_b.dot(theta) - y) + l1 * np.sign(theta) + 2 * l2 * theta
        )
        theta = theta - eta * gradients
        path.append(theta)
    return np.array(path)


fig, axes = plt.subplots(ncols=2, figsize=(10, 4), sharey=True)

for i, N, l1, l2, title in ((0, N1, 2.0, 0, "Lasso"), (1, N2, 0, 2.0, "Ridge")):
    JR = J + l1 * N1 + l2 * N2**2

    tr_min_idx = np.argmin(JR)
    tr_max_idx = np.argmax(JR)

    levels = (np.exp(np.linspace(0, 1, 20)) - 1) * (np.max(JR) - np.min(JR)) + np.min(
        JR
    )
    levelsJ = (np.exp(np.linspace(0, 1, 20)) - 1) * (np.max(J) - np.min(J)) + np.min(J)
    levelsJR = (np.exp(np.linspace(0, 1, 20)) - 1) * (np.max(JR) - np.min(JR)) + np.min(
        JR
    )
    levelsN = np.linspace(0, np.max(N), 10)

    path_J = bdg_path(np.array([2.0, 0.5]).reshape(-1, 1), X_b, y, l1=0, l2=0)
    path_JR = bdg_path(np.array([2.0, 0.5]).reshape(-1, 1), X_b, y, l1, l2)
    path_N = bdg_path(
        np.array([2.0, 0.5]).reshape(-1, 1),
        X_b,
        y,
        np.sign(l1) / 3,
        np.sign(l2),
        theta=0.1,
    )

    ax = axes[i, 0]
    ax.grid(True)
    ax.axhline(y=0, color="k")
    ax.axvline(x=0, color="k")
    ax.contourf(T[0], T[1], J.reshape(T[0].shape), levels=levelsJ, alpha=0.9)
    ax.contour(T[0], T[1], N.reshape(T[0].shape), levels=levelsN)
    ax.plot(path_J[:, 0], path_J[:, 1], "w-o")
    ax.plot(path_N[:, 0], path_N[:, 1], "y-^")
    ax.plot(path_JR[:, 0], path_JR[:, 1], "r-s")
    ax.plot(0, 0, "ys")
    ax.plot(t_min_idx, 0, "ys")
    ax.plot(t_max_idx, 0, "ys")
    ax.plot(t_min_idx, JR.ravel()[t_min_idx], "ys")
    ax.plot(t_max_idx, JR.ravel()[t_max_idx], "ys")
    ax.set_title(r"$\ell_{}$ penalty".format(i + 1), fontsize=16)
    ax.axis([t1a, t1b, t2a, t2b])
    if i == 1:
        ax.set_xlabel(r"$\theta_1$", fontsize=16)
    ax.set_ylabel(r"$\theta_2$", fontsize=16, rotation=0)

    ax = axes[i, 1]
    ax.grid(True)
    ax.axhline(y=0, color="k")
    ax.axvline(x=0, color="k")
    ax.contourf(T[0], T[1], JR.reshape(T[0].shape), levels=levelsJR, alpha=0.9)
    ax.plot(path_J[:, 0], path_J[:, 1], "w-o")
    ax.plot(path_N[:, 0], path_N[:, 1], "y-^")
    ax.plot(path_JR[:, 0], path_JR[:, 1], "r-s")
    ax.plot(0, 0, "ys")
    ax.plot(t_min_idx, 0, "ys")
    ax.plot(t_max_idx, 0, "ys")
    ax.plot(t_min_idx, JR.ravel()[t_min_idx], "ys")
    ax.plot(t_max_idx, JR.ravel()[t_max_idx], "ys")
    ax.set_title(title, fontsize=16)
    ax.axis([t1a, t1b, t2a, t2b])
    if i == 1:
        ax.set_xlabel(r"$\theta_1$", fontsize=16)

plt.show()
save_fig("lasso_vs_ridge_plot.png")

# Elastic Net

from sklearn.linear_model import ElasticNet

elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
elastic_net.fit(X, y)
print(elastic_net.predict([[1.5]]))

# Early Stopping

from copy import deepcopy
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# prepare the data

np.random.seed(42)
m = 100
X = 6 * np.random.rand(m, 1) - 3
y = 2 + X + 0.5 * X**2 + np.random.randn(m, 1)

X_train, X_val, y_train, y_val = train_test_split(
    X[:50], y[:50].ravel(), test_size=0.5, random_state=10
)

preprocessing = make_pipeline(
    PolynomialFeatures(degree=90, include_bias=False), StandardScaler()
)
X_train_prep = preprocessing.fit_transform(X_train)
X_val_prep = preprocessing.transform(X_val)
sdg_reg = SGDRegressor(
    max_iter=1,
    tol=-np.infty,
    warm_start=True,
    penalty=None,
    learning_rate="constant",
    eta0=0.0005,
    random_state=42,
)
n_epochs = 500
best_valid_rmse = float("inf")
train_errors, val_errors = [], []

for epoch in range(n_epochs):
    sdg_reg.fit(X_train_prep, y_train)
    y_train_predict = sdg_reg.predict(X_train_prep)
    y_val_predict = sdg_reg.predict(X_val_prep)
    train_errors.append(mean_squared_error(y_train, y_train_predict))
    val_errors.append(mean_squared_error(y_val, y_val_predict))
    if val_errors[-1] < best_valid_rmse:
        best_valid_rmse = val_errors[-1]
        best_model = deepcopy(sdg_reg)

y_train_predict = best_model.predict(X_train_prep)
train_error = mean_squared_error(y_train, y_train_predict)
val_errors.append(mean_squared_error(y_val, y_val_predict))
train_errors.append(train_error)

best_epoch = np.argmin(val_errors)
best_val_rmse = np.sqrt(val_errors[best_epoch])
plt.figure(figsize=(10, 4))
plt.subplot("Best Model", fontsize=14)
plt.plot([0, n_epochs], [best_val_rmse, best_val_rmse], "k:", linewidth=2)
plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="Validation set")
plt.plot(np.sqrt(train_errors), "r--", linewidth=2, label="Training set")
plt.legend(loc="upper right", fontsize=14)
plt.xlabel("Epoch", fontsize=14)
plt.ylabel("RMSE", fontsize=14)
plt.show()

save_fig("early_stopping_plot.png")

# Logistic Regression

# Estimating probabilities

lim = 5
t = np.linspace(-lim, lim, 100)
sig = 1 / (1 + np.exp(-t))

plt.figure(figsize=(9, 3))
plt.subplot(121)
plt.plot([-lim, lim], [0, 0], "k-")
plt.plot([-lim, lim], [0.5, 0.5], "k:")
plt.plot([-lim, lim], [1, 1], "k:")
plt.plot([0, 0], [-1.1, 1.1], "k-")
plt.plot(t, sig, "b-", linewidth=2, label=r"$\sigma(t) = \frac{1}{1 + e^{-t}}$")
plt.xlabel("t")
plt.legend(loc="upper left", fontsize=20)
plt.axis([-lim, lim, -0.1, 1.1])

plt.subplot(122)
plt.plot([-lim, lim], [0, 1], "k-")
plt.plot([-lim, lim], [0.5, 0.5], "k:")
plt.plot([0, 0], [-0.1, 1.1], "k-")
plt.plot(t, sig, "b-", linewidth=2, label=r"$\hat{p} = h_{\theta}(x)$")
plt.plot(t, np.zeros(t.shape), "r-", linewidth=2, label="Prediction y=0")
plt.plot(t, np.ones(t.shape), "g--", linewidth=2, label="Prediction y=1")
plt.xlabel("t")

plt.legend(loc="center left", fontsize=20)
plt.axis([-lim, lim, -0.1, 1.1])
plt.show()

# Decision Boundaries

from sklearn import datasets
from sklearn.datasets import load_iris

iris = datasets.load_iris()
list(iris.keys())

X = iris["data"][:, 3:]
y = (iris["target"] == 2).astype(np.int)

print(iris.DESCR)

print(list(iris.keys()))

print(iris["target_names"])

print(iris["feature_names"])

print(iris["data"])

print(iris["target"])

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

log_reg = LogisticRegression(solver="liblinear", random_state=42)
log_reg.fit(X, y)

X = iris.data[["petal width (cm)"]].values
y = (iris.target == 2).astype(np.int)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

log_reg = LogisticRegression(solver="liblinear", random_state=42)
log_reg.fit(X_train, y_train)

X_new = np.linspace(0, 3, 1000).reshape(-1, 1)
y_proba = log_reg.predict_proba(X_new)
decision_boundary = X_new[y_proba[:, 1] >= 0.5][0]

plt.figure(figsize=(8, 3))  # extra code – not needed, just formatting
plt.plot(X_new, y_proba[:, 0], "b--", linewidth=2, label="Not Iris virginica proba")
plt.plot(X_new, y_proba[:, 1], "g-", linewidth=2, label="Iris virginica proba")
plt.plot(
    [decision_boundary, decision_boundary],
    [0, 1],
    "k:",
    linewidth=2,
    label="Decision boundary",
)

# extra code – this section beautifies and saves Figure 4–23
plt.arrow(
    x=decision_boundary,
    y=0.08,
    dx=-0.3,
    dy=0,
    head_width=0.05,
    head_length=0.1,
    fc="b",
    ec="b",
)
plt.arrow(
    x=decision_boundary,
    y=0.92,
    dx=0.3,
    dy=0,
    head_width=0.05,
    head_length=0.1,
    fc="g",
    ec="g",
)
plt.plot(X_train[y_train == 0], y_train[y_train == 0], "bs")
plt.plot(X_train[y_train == 1], y_train[y_train == 1], "g^")
plt.xlabel("Petal width (cm)")
plt.ylabel("Probability")
plt.legend(loc="center left")
plt.axis([0, 3, -0.02, 1.02])
plt.grid()
save_fig("logistic_regression_plot")

plt.show()

print(decision_boundary)

log_reg.predict([[1.7], [1.5]])

X = iris["data"][:, (2, 3)]  # petal length, petal width
y = iris.target_names[iris.target] == "virginica"
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

log_reg = LogisticRegression(solver="liblinear", C=10**10, random_state=42)
log_reg.fit(X_train, y_train)

x0, x1 = np.meshgrid(
    np.linspace(2.9, 7, 500).reshape(-1, 1),
    np.linspace(0.8, 2.7, 200).reshape(-1, 1),
)

X_new = np.c_[x0.ravel(), x1.ravel()]
y_proba = log_reg.predict_proba(X_new)
zz = y_proba[:, 1].reshape(x0.shape)

left_right = np.array([2.9, 7])
boundary = (
    -(log_reg.coef_[0][0] * left_right + log_reg.intercept_[0]) / log_reg.coef_[0][1]
)

plt.figure(figsize=(10, 4))
plt.plot(X_train[y_train == 0, 0], X_train[y_train == 0, 1], "bs")
plt.plot(X_train[y_train == 1, 0], X_train[y_train == 1, 1], "g^")
contour = plt.contour(x0, x1, zz, cmap=plt.cm.brg)
plt.clabel(contour, inline=1)
plt.plot(left_right, boundary, "k--", linewidth=3)
plt.text(3.5, 1.27, "Not Iris virginica", color="b", ha="center")
plt.text(6.5, 2.3, "Iris virginica", color="g", ha="center")
plt.xlabel("Petal length")
plt.ylabel("Petal width")
plt.axis([2.9, 7, 0.8, 2.7])
plt.grid()
save_fig("logistic_regression_contour_plot")
plt.show()

# Softmax regression

X = iris.data[["petal length (cm)", "petal width (cm)"]].values 
y = iris["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

softmax_reg = LogisticRegression(multi_class="multinomial", solver="lbfgs", C=10, random_state=42)
softmax_reg.fit(X_train, y_train)

softmax_reg.predict([[5, 2]])

softmax_reg.predict_proba([[5, 2]])

from matplotlib.colors import ListedColormap

custom_cmap = ListedColormap(["#fafab0", "#9898ff", "#a0faa0"])

x0, x1 = np.meshgrid(np.linspace(0, 8, 500).reshape(-1, 1), np.linspace(0, 3.5, 200).reshape(-1, 1))

X_new = np.c_[x0.ravel(), x1.ravel()]

y_proba = softmax_reg.predict_proba(X_new)
y_predict = softmax_reg.predict(X_new)

zz1 = y_proba[:, 1].reshape(x0.shape)
zz = y_predict.reshape(x0.shape)

plt.figure(figsize=(10, 4))
plt.plot(X[y==2, 0], X[y==2, 1], "g^", label="Iris virginica")
plt.plot(X[y==1, 0], X[y==1, 1], "bs", label="Iris versicolor")
plt.plot(X[y==0, 0], X[y==0, 1], "yo", label="Iris setosa")

plt.contourf(x0, x1, zz, cmap=custom_cmap)
contour = plt.contour(x0, x1, zz1, cmap=plt.cm.brg)

plt.clabel(contour, inline=1, fontsize=12)
plt.xlabel("Petal length", fontsize=14)
plt.ylabel("Petal width", fontsize=14)
plt.legend(loc="center left", fontsize=14)
plt.axis([0, 7, 0, 3.5])
plt.grid(True)
save_fig("softmax_regression_contour_plot")
plt.show()

softmax_reg.predict([[5, 2]])

softmax_reg.predict_proba([[5, 2]])

# Exercise solutions

# 1. to 11.

# See appendix A.

# If you have millions of features, you can try PCA, preserving enough variance to
# obtain a reasonable dataset with fewer features (e.g., 200), then train your

# If the training set have different scales, the best approach is to 
# scale the data using the StandardScaler, then train the model.

# Can a gradient descent get stuck in a local minimum when training a Logistic Regression model?
# No, because the cost function is convex. You can prove that the cost function is convex by computing its Hessian matrix and showing that it is always positive definite.

# Do all gradient descent algorithms lead to the same model provided you let them run long enough?
# If they use the same learning rate and you wait long enough, then they will all approach the same solution, but they will never be exactly the same.

# Suppose you use batch gradient descent and you plot the validation error at every epoch. If you notice that the validation error consistently goes up, what is likely going on? How can you fix this?
# If the validation error consistently goes up after every epoch, then one possibility is that the learning rate is too high and the algorithm is diverging. If the training error also goes up, then this is clearly the problem and you should reduce the learning rate. However, if the training error is not going up, then your model is overfitting the training set and you should stop training.

# Is it a good idea to stop Mini-batch Gradient Descent immediately when the validation error goes up?
# No, it is normal for the validation error to go up and down, so you should not stop right away. You should only stop when the validation error has been above the minimum for some time (when you are confident that the model will not do any better).

# Which Gradient Descent algorithm (among those we discussed) will reach the vicinity of the optimal solution the fastest? Which will actually converge? How can you make the others converge as well?
# Stochastic Gradient Descent has the fastest training iteration since it considers only one training instance at a time, so it is generally the first to reach the vicinity of the global optimum (or Mini-batch Gradient Descent with a very small mini-batch size). However, only Batch Gradient Descent will actually converge, given enough training time. As mentioned, Stochastic GD and Mini-batch GD will bounce around the optimum, unless you gradually reduce the learning rate.

# Suppose you are using Polynomial Regression. You plot the learning curves and you notice that there is a large gap between the training error and the validation error. What is happening? What are three ways to solve this?
# If the validation error is much higher than the training error, this is likely because your model is overfitting the training set. One way to try to fix this is to reduce the polynomial degree: a model with fewer degrees of freedom is less likely to overfit. Another thing you can try is to regularize the model—for example, by adding an ℓ2 penalty (Ridge) or an ℓ1 penalty (Lasso) to the cost function. This will also reduce the degrees of freedom of the model. Lastly, you can try to increase the size of the training set.

# Suppose you are using Ridge Regression and you notice that the training error and the validation error are almost equal and fairly high. Would you say that the model suffers from high bias or high variance? Should you increase the regularization hyperparameter α or reduce it?
# If both the training error and the validation error are almost equal and fairly high, the model is likely underfitting the training set, which means it has a high bias. You should try reducing the regularization hyperparameter α.

# Why would you want to use:
# Ridge Regression instead of plain Linear Regression (i.e., without any regularization)?
# Lasso instead of Ridge Regression?
# Elastic Net instead of Lasso?
# Plain Linear Regression is the baseline model, but if you suspect that some features are useless, you should prefer Ridge Regression over plain Linear Regression. Lasso Regression is a good default if you suspect that only a few features are useful. In general, Elastic Net is preferred over Lasso since Lasso may behave erratically when the number of features is greater than the number of training instances or when several features are strongly correlated.
# If you want to have a model that you can interpret, and you suspect that only a few features are useful, you should use Lasso since it tends to reduce the useless features’ weights down to zero, as we have discussed. If you suspect that there is some useless features, but you are not sure, you should prefer Ridge Regression over Lasso since it will reduce the useless features’ weights but not to zero. If you want to have a model that you can interpret, and you suspect that the features may be correlated, you should use Elastic Net since Lasso may behave erratically in this case.

# Suppose you want to classify pictures as outdoor/indoor and daytime/nighttime. Should you implement two Logistic Regression classifiers or one Softmax Regression classifier?
# If you want to have two binary classifiers (e.g., for simplicity, or because you want to be able to get the precision and recall of each class), you should train two Logistic Regression classifiers. If you want a classifier that can output multiple classes, you should train a Softmax Regression classifier.

# Implement Batch Gradient Descent with early stopping for Softmax Regression (without using Scikit-Learn).

# Load the data

X = iris["data"][:, (2, 3)]  # petal length, petal width
y = iris["target"]

# Add the bias term for every instance ($x_0 = 1$)

X_with_bias = np.c_[np.ones([len(X), 1]), X]

# Set the random seed so that the output of this exercise solution is reproducible.

np.random.seed(2042)

# Split the dataset into a training set and a test set

test_ratio = 0.2

validation_ratio = 0.2

total_size = len(X_with_bias)

test_size = int(total_size * test_ratio)

validation_size = int(total_size * validation_ratio)

train_size = total_size - test_size - validation_size

rnd_indices = np.random.permutation(total_size)

X_train = X_with_bias[rnd_indices[:train_size]]

y_train = y[rnd_indices[:train_size]]

X_valid = X_with_bias[rnd_indices[train_size:-test_size]]

y_valid = y[rnd_indices[train_size:-test_size]]

X_test = X_with_bias[rnd_indices[-test_size:]]

y_test = y[rnd_indices[-test_size:]]

# One Hot Encoding

def to_one_hot(y):    
        n_classes = y.max() + 1    
        m = len(y)    
        Y_one_hot = np.zeros((m, n_classes))    
        Y_one_hot[np.arange(m), y] = 1    
        return Y_one_hot
    
y_train_one_hot = to_one_hot(y_train)

y_valid_one_hot = to_one_hot(y_valid)

y_test_one_hot = to_one_hot(y_test)

# Softmax function

def softmax(logits):
    exps = np.exp(logits)
    exp_sums = np.sum(exps, axis=1, keepdims=True)
    return exps / exp_sums

n_inputs = X_train.shape[1] # == 3 (2 features plus the bias term)

n_outputs = len(np.unique(y_train))   # == 3 (3 iris classes)

# Training

eta = 0.01

n_iterations = 5001

m = len(X_train)

epsilon = 1e-7

Theta = np.random.randn(n_inputs, n_outputs)

for iteration in range(n_iterations):
    logits = X_train.dot(Theta)
    Y_proba = softmax(logits)
    loss = -np.mean(np.sum(y_train_one_hot * np.log(Y_proba + epsilon), axis=1))
    error = Y_proba - y_train_one_hot
    if iteration % 500 == 0:
        print(iteration, loss)
    gradients = 1/m * X_train.T.dot(error)
    Theta = Theta - eta * gradients
    
# Validation

logits = X_valid.dot(Theta)

Y_proba = softmax(logits)

y_predict = np.argmax(Y_proba, axis=1)

accuracy_score = np.mean(y_predict == y_valid)

print(accuracy_score)

# Add early stopping

eta = 0.1

n_iterations = 5001

m = len(X_train)

epsilon = 1e-7

best_loss = np.infty

Theta = np.random.randn(n_inputs, n_outputs)

for iteration in range(n_iterations):
    logits = X_train.dot(Theta)
    Y_proba = softmax(logits)
    error = Y_proba - y_train_one_hot
    gradients = 1/m * X_train.T.dot(error)
    Theta = Theta - eta * gradients
    logits = X_valid.dot(Theta)
    Y_proba = softmax(logits)
    loss = -np.mean(np.sum(y_valid_one_hot * np.log(Y_proba + epsilon), axis=1))
    if iteration % 500 == 0:
        print(iteration, loss)
    if loss < best_loss:
        best_loss = loss
    else:
        print(iteration - 1, best_loss)
        print(iteration, loss, "early stopping!")
        break
    
# Predictions on the test set

logits = X_valid.dot(Theta)

Y_proba = softmax(logits)

y_predict = np.argmax(Y_proba, axis=1)

accuracy_score = np.mean(y_predict == y_valid)

print(accuracy_score)

# Plot the model's predictions on the whole dataset

x0, x1 = np.meshgrid(
        np.linspace(0, 8, 500).reshape(-1, 1),
        np.linspace(0, 3.5, 200).reshape(-1, 1))
X_new = np.c_[x0.ravel(), x1.ravel()]
X_new_with_bias = np.c_[np.ones([len(X_new), 1]), X_new]
y_proba = softmax(X_new_with_bias.dot(Theta))
y_predict = np.argmax(y_proba, axis=1)

zz1 = y_proba[:, 1].reshape(x0.shape)
zz = y_predict.reshape(x0.shape)

plt.figure(figsize=(10, 4))
plt.plot(X[y==2, 0], X[y==2, 1], "g^", label="Iris virginica")
plt.plot(X[y==1, 0], X[y==1, 1], "bs", label="Iris versicolor")
plt.plot(X[y==0, 0], X[y==0, 1], "yo", label="Iris setosa")

from matplotlib.colors import ListedColormap

custom_cmap = ListedColormap(['#fafab0','#9898ff','#a0faa0'])

plt.contourf(x0, x1, zz, cmap=custom_cmap)
contour = plt.contour(x0, x1, zz1, cmap=plt.cm.brg)

plt.clabel(contour, inline=1, fontsize=12)
plt.xlabel("Petal length", fontsize=14)
plt.ylabel("Petal width", fontsize=14)
plt.legend(loc="center left", fontsize=14)
plt.axis([0, 7, 0, 3.5])
plt.show()
save_figure("softmax_regression_contour_plot")

# Using Scikit-Learn

X = iris["data"][:, (2, 3)]  # petal length, petal width

y = iris["target"]

softmax_reg = LogisticRegression(multi_class="multinomial",solver="lbfgs", C=10)

softmax_reg.fit(X, y)

softmax_reg.predict([[5, 2]])

softmax_reg.predict_proba([[5, 2]])

# Exercise 12

X = iris["data"][:, (2, 3)]  # petal length, petal width

y = iris["target"]

softmax_reg = LogisticRegression(multi_class="multinomial",solver="lbfgs", C=10)

softmax_reg.fit(X, y)

x0, x1 = np.meshgrid((np.linspace(0, 8, 500).reshape(-1, 1)),(np.linspace(0, 3.5, 200).reshape(-1, 1)))

X_new = np.c_[x0.ravel(), x1.ravel()]

y_proba = softmax_reg.predict_proba(X_new)

y_predict = softmax_reg.predict(X_new)

zz1 = y_proba[:, 1].reshape(x0.shape)

zz = y_predict.reshape(x0.shape)

plt.figure(figsize=(10, 4))

plt.plot(X[y==2, 0], X[y==2, 1], "g^", label="Iris virginica")
plt.plot(X[y==1, 0], X[y==1, 1], "bs", label="Iris versicolor")
plt.plot(X[y==0, 0], X[y==0, 1], "yo", label="Iris setosa")

from matplotlib.colors import ListedColormap

custom_cmap = ListedColormap(['#fafab0','#9898ff','#a0faa0'])

plt.contourf(x0, x1, zz, cmap=custom_cmap)

contour = plt.contour(x0, x1, zz1, cmap=plt.cm.brg)

plt.clabel(contour, inline=1, fontsize=12)
plt.xlabel("Petal length", fontsize=14)
plt.ylabel("Petal width", fontsize=14)

plt.legend(loc="center left", fontsize=14)
plt.axis([0, 7, 0, 3.5])
plt.grid(True)
plt.show()

save_figure("softmax_regression_contour_plot")

# Exercise 13

X = iris["data"][:, (2, 3)]  # petal length, petal width

y = iris["target"]

softmax_reg = LogisticRegression(multi_class="multinomial",solver="lbfgs", C=10)

softmax_reg.fit(X, y)

x0, x1 = np.meshgrid((np.linspace(0, 8, 500).reshape(-1, 1)),(np.linspace(0, 3.5, 200).reshape(-1, 1)))

X_new = np.c_[x0.ravel(), x1.ravel()]

y_proba = softmax_reg.predict_proba(X_new)

y_predict = softmax_reg.predict(X_new)

zz1 = y_proba[:, 1].reshape(x0.shape)

zz = y_predict.reshape(x0.shape)

left_right = np.array([2.9, 7])

boundary = -(softmax_reg.coef_[0][0] * left_right + softmax_reg.intercept_[0]) / softmax_reg.coef_[0][1]

plt.figure(figsize=(10, 4))

plt.plot(X[y==2, 0], X[y==2, 1], "g^", label="Iris virginica")

plt.plot(X[y==1, 0], X[y==1, 1], "bs", label="Iris versicolor")

plt.plot(X[y==0, 0], X[y==0, 1], "yo", label="Iris setosa")

plt.plot(left_right, boundary, "k--", linewidth=3)

plt.text(3.5, 1.5, "Not Iris virginica", fontsize=14, color="b", ha="center")

plt.text(6.5, 2.3, "Iris virginica", fontsize=14, color="g", ha="center")

plt.xlabel("Petal length", fontsize=14)

plt.ylabel("Petal width", fontsize=14)

plt.legend(loc="center left", fontsize=14)

plt.axis([0, 7, 0, 3.5])

plt.grid(True)

plt.show()

save_figure("softmax_regression_contour_plot")

# Exercise 14

X = iris["data"][:, (2, 3)]  # petal length, petal width

y = iris["target"]

softmax_reg = LogisticRegression(multi_class="multinomial",solver="lbfgs", C=10)

softmax_reg.fit(X, y)

x0, x1 = np.meshgrid((np.linspace(0, 8, 500).reshape(-1, 1)),(np.linspace(0, 3.5, 200).reshape(-1, 1)))

X_new = np.c_[x0.ravel(), x1.ravel()]

y_proba = softmax_reg.predict_proba(X_new)

y_predict = softmax_reg.predict(X_new)

zz1 = y_proba[:, 1].reshape(x0.shape)

zz = y_predict.reshape(x0.shape)

left_right = np.array([2.9, 7])

boundary = -(softmax_reg.coef_[0][0] * left_right + softmax_reg.intercept_[0]) / softmax_reg.coef_[0][1]

plt.figure(figsize=(10, 4))

plt.plot(X[y==2, 0], X[y==2, 1], "g^", label="Iris virginica")

plt.plot(X[y==1, 0], X[y==1, 1], "bs", label="Iris versicolor")

plt.plot(X[y==0, 0], X[y==0, 1], "yo", label="Iris setosa")

plt.plot(left_right, boundary, "k--", linewidth=3)

plt.text(3.5, 1.5, "Not Iris virginica", fontsize=14, color="b", ha="center")

plt.text(6.5, 2.3, "Iris virginica", fontsize=14, color="g", ha="center")

plt.xlabel("Petal length", fontsize=14)

plt.ylabel("Petal width", fontsize=14)

plt.legend(loc="center left", fontsize=14)

plt.axis([0, 7, 0, 3.5])

plt.grid(True)

plt.show()

save_figure("softmax_regression_contour_plot")




