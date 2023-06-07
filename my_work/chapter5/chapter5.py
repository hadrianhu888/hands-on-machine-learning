import sys

assert sys.version_info >= (3, 7)  # make sure we have Python 3.4+

from packaging import version
import sklearn

assert version.parse(sklearn.__version__) >= version.parse("1.0.1")

import matplotlib.pyplot as plt

plt.rc("font", size=14)
plt.rc("axes", labelsize=14, titlesize=14)
plt.rc("legend", fontsize=14)
plt.rc("xtick", labelsize=10)
plt.rc("ytick", labelsize=10)

from pathlib import Path

IMAGES_PATH = Path() / "images" / "svm"
IMAGES_PATH.mkdir(parents=True, exist_ok=True)


def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = IMAGES_PATH / f"{fig_id}.{fig_extension}"
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


# linear svm classification

import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC
from sklearn import datasets

iris = datasets.load_iris(as_frame=True)
X = iris.data[["petal length (cm)", "petal width (cm)"]].values
y = iris.target

setosa_or_versicolor = (y == 0) | (y == 1)
X = X[setosa_or_versicolor]
y = y[setosa_or_versicolor]

# SVM Classifier model
svm_clf = SVC(kernel="linear", C=1e9)
svm_clf.fit(X, y)

# Bad models
x0 = np.linspace(0, 5.5, 200)
pred_1 = 5 * x0 - 20
pred_2 = x0 - 1.8
pred_3 = 0.1 * x0 + 0.5


def plot_svc_decision_boundary(svm_clf, xmin, xmax):
    w = svm_clf.coef_[0]
    b = svm_clf.intercept_[0]

    # At the decision boundary, w0*x0 + w1*x1 + b = 0
    # => x1 = -w0/w1 * x0 - b/w1
    x0 = np.linspace(xmin, xmax, 200)
    decision_boundary = -w[0] / w[1] * x0 - b / w[1]

    margin = 1 / w[1]
    gutter_up = decision_boundary + margin
    gutter_down = decision_boundary - margin
    svs = svm_clf.support_vectors_

    plt.plot(x0, decision_boundary, "k-", linewidth=2, zorder=-2)
    plt.plot(x0, gutter_up, "k--", linewidth=2, zorder=-2)
    plt.plot(x0, gutter_down, "k--", linewidth=2, zorder=-2)
    plt.scatter(svs[:, 0], svs[:, 1], s=180, facecolors="#AAA", zorder=-1)


fig, axes = plt.subplots(ncols=2, figsize=(10, 2.7), sharey=True)

plt.sca(axes[0])
plt.plot(x0, pred_1, "g--", linewidth=2)
plt.plot(x0, pred_2, "m-", linewidth=2)
plt.plot(x0, pred_3, "r-", linewidth=2)
plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], "bs", label="Iris versicolor")
plt.plot(X[:, 0][y == 0], X[:, 1][y == 0], "yo", label="Iris setosa")
plt.xlabel("Petal length", fontsize=14)
plt.ylabel("Petal width", fontsize=14)
plt.legend(loc="upper left", fontsize=14)
plt.axis([4, 6, 0.8, 2.8])
plt.gca().set_aspect("equal", adjustable="box")
plt.grid()

plt.sca(axes[1])
plot_svc_decision_boundary(svm_clf, 0, 5.5)
plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], "bs")
plt.plot(X[:, 0][y == 0], X[:, 1][y == 0], "yo")
plt.xlabel("Petal length")
plt.axis([0, 5.5, 0, 2])
plt.gca().set_aspect("equal")
plt.grid()

save_fig("large_margin_classification_plot")
plt.show()

# extra code - this cell generates and saves Figure 5-2

from sklearn.preprocessing import StandardScaler

Xs = np.array([[1, 50], [5, 20], [3, 80], [5, 60]]).astype(np.float64)
ys = np.array([0, 0, 1, 1])
svm_clf = SVC(kernel="linear", C=100).fit(Xs, ys)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(Xs)
svm_clf_scaled = SVC(kernel="linear", C=100).fit(X_scaled, ys)

plt.figure(figsize=(9, 2.7))
plt.subplot(121)
plt.plot(Xs[:, 0][ys == 1], Xs[:, 1][ys == 1], "bo")
plt.plot(Xs[:, 0][ys == 0], Xs[:, 1][ys == 0], "ms")
plot_svc_decision_boundary(svm_clf, 0, 6)
plt.xlabel("$x_0$")
plt.ylabel("$x_1$    ", rotation=0)
plt.title("Unscaled")
plt.axis([0, 6, 0, 90])
plt.grid()

plt.subplot(122)
plt.plot(X_scaled[:, 0][ys == 1], X_scaled[:, 1][ys == 1], "bo")
plt.plot(X_scaled[:, 0][ys == 0], X_scaled[:, 1][ys == 0], "ms")
plot_svc_decision_boundary(svm_clf_scaled, -2, 2)
plt.xlabel("$x'_0$")
plt.ylabel("$x'_1$  ", rotation=0)
plt.title("Scaled")
plt.axis([-2, 2, -2, 2])
plt.grid()

save_fig("sensitivity_to_feature_scales_plot")
plt.show()

# soft margin classification

X_outliers = np.array([[3.4, 1.3], [3.4, 1.0]])
y_outliers = np.array([0, 0])

X_o1 = np.concatenate([X, X_outliers[:1]], axis=0)
yo1 = np.concatenate([y, y_outliers[:1]], axis=0)
Xo2 = np.concatenate([X, X_outliers[1:]], axis=0)
yo2 = np.concatenate([y, y_outliers[1:]], axis=0)

svm_clf2 = SVC(kernel="linear", C=10**9)
svm_clf2.fit(Xo2, yo2)

fig, axes = plt.subplots(ncols=2, figsize=(10, 2.7), sharey=True)

Xo1 = np.concatenate([X[:1], X_outliers[:1]], axis=0)
yo1 = np.concatenate([y[:1], y_outliers[:1]], axis=0)

plt.sca(axes[0])
plt.plot(Xo1[:, 0][yo1 == 1], Xo1[:, 1][yo1 == 1], "bs")
plt.plot(Xo1[:, 0][yo1 == 0], Xo1[:, 1][yo1 == 0], "yo")
plt.text(0.3, 1.0, "Impossible!", color="red", fontsize=18)
plt.xlabel("Petal length")
plt.ylabel("Petal width")
plt.annotate(
    "Outlier",
    xy=(X_outliers[0][0], X_outliers[0][1]),
    xytext=(2.5, 1.7),
    ha="center",
    arrowprops=dict(facecolor="black", shrink=0.1),
)
plt.axis([0, 5.5, 0, 2])
plt.grid()

plt.sca(axes[1])
plt.plot(Xo2[:, 0][yo2 == 1], Xo2[:, 1][yo2 == 1], "bs")
plt.plot(Xo2[:, 0][yo2 == 0], Xo2[:, 1][yo2 == 0], "yo")
plot_svc_decision_boundary(svm_clf2, 0, 5.5)
plt.xlabel("Petal length")
plt.annotate(
    "Outlier",
    xy=(X_outliers[1][0], X_outliers[1][1]),
    xytext=(3.2, 0.08),
    ha="center",
    arrowprops=dict(facecolor="black", shrink=0.1),
)
plt.axis([0, 5.5, 0, 2])
plt.grid()

save_fig("sensitivity_to_outliers_plot")
plt.show()

import numpy as np
from sklearn.datasets import load_iris
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


def make_pipeline(scaler, svm_clf):
    return Pipeline([("scaler", scaler), ("linear_svc", svm_clf)])


iris = load_iris(as_frame=True)
X = iris.data[["petal length (cm)", "petal width (cm)"]]
y = iris.target == 2

svm_clf = make_pipeline(StandardScaler(), LinearSVC(C=1, loss="hinge", random_state=42))

svm_clf.fit(X, y)

X_new = [[5.5, 1.7], [5.0, 1.5]]
svm_clf.predict(X_new)

svm_clf.decision_function(X_new)

# extra code – this cell generates and saves Figure 5–4

scaler = StandardScaler()
svm_clf1 = LinearSVC(C=1, max_iter=10_000, random_state=42)
svm_clf2 = LinearSVC(C=100, max_iter=10_000, random_state=42)

scaled_svm_clf1 = make_pipeline(scaler, svm_clf1)
scaled_svm_clf2 = make_pipeline(scaler, svm_clf2)

scaled_svm_clf1.fit(X, y)
scaled_svm_clf2.fit(X, y)

# Convert to unscaled parameters
b1 = svm_clf1.decision_function([-scaler.mean_ / scaler.scale_])
b2 = svm_clf2.decision_function([-scaler.mean_ / scaler.scale_])
w1 = svm_clf1.coef_[0] / scaler.scale_
w2 = svm_clf2.coef_[0] / scaler.scale_
svm_clf1.intercept_ = np.array([b1])
svm_clf2.intercept_ = np.array([b2])
svm_clf1.coef_ = np.array([w1])
svm_clf2.coef_ = np.array([w2])

# Find support vectors (LinearSVC does not do this automatically)
t = y * 2 - 1
support_vectors_idx1 = (t * (X.dot(w1) + b1) < 1).ravel()
support_vectors_idx2 = (t * (X.dot(w2) + b2) < 1).ravel()
svm_clf1.support_vectors_ = X[support_vectors_idx1]
svm_clf2.support_vectors_ = X[support_vectors_idx2]

fig, axes = plt.subplots(ncols=2, figsize=(10, 2.7), sharey=True)

plt.sca(axes[0])
plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], "g^", label="Iris virginica")
plt.plot(X[:, 0][y == 0], X[:, 1][y == 0], "bs", label="Iris versicolor")
plot_svc_decision_boundary(svm_clf1, 4, 5.9)
plt.xlabel("Petal length")
plt.ylabel("Petal width")
plt.legend(loc="upper left")
plt.title(f"$C = {svm_clf1.C}$")
plt.axis([4, 5.9, 0.8, 2.8])
plt.grid()

plt.sca(axes[1])
plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], "g^")
plt.plot(X[:, 0][y == 0], X[:, 1][y == 0], "bs")
plot_svc_decision_boundary(svm_clf2, 4, 5.99)
plt.xlabel("Petal length")
plt.title(f"$C = {svm_clf2.C}$")
plt.axis([4, 5.9, 0.8, 2.8])
plt.grid()

save_fig("regularization_plot")
plt.show()

# Nonlinear SVM Classification

X1D = np.linspace(-4, 4, 9).reshape(-1, 1)
X2D = np.c_[X1D, X1D**2]

y = np.array([0, 0, 1, 1, 1, 1, 1, 0, 0])

plt.figure(figsize=(10, 3))

plt.subplot(121)
plt.grid(True)
plt.axhline(y=0, color="k")
plt.plot(X1D[:, 0][y == 0], np.zeros(4), "bs")
plt.plot(X1D[:, 0][y == 1], np.zeros(5), "g^")
plt.gca().get_yaxis().set_ticks([])
plt.xlabel("$x_1$")
plt.axis([-4.5, 4.5, -0.2, 0.2])

plt.subplot(122)
plt.grid(True)
plt.axhline(y=0, color="k")
plt.axvline(x=0, color="k")
plt.plot(X2D[:, 0][y == 0], X2D[:, 1][y == 0], "bs")
plt.plot(X2D[:, 0][y == 1], X2D[:, 1][y == 1], "g^")
plt.xlabel("$x_1$")
plt.ylabel("$x_2$  ", rotation=0)
plt.gca().get_yaxis().set_ticks([0, 4, 8, 12, 16])
plt.plot([-4.5, 4.5], [6.5, 6.5], "r--", linewidth=3)
plt.axis([-4.5, 4.5, -1, 17])

plt.subplots_adjust(right=1)

save_fig("higher_dimensions_plot", tight_layout=False)
plt.show()

from sklearn.datasets import make_moons
from sklearn.preprocessing import PolynomialFeatures

X, y = make_moons(n_samples=100, noise=0.15, random_state=42)

polynomial_svm_clf = make_pipeline(
    PolynomialFeatures(degree=3),
    StandardScaler(),
    LinearSVC(C=10, loss="hinge", random_state=42),
)

polynomial_svm_clf.fit(X, y)


def plot_dataset(X, y, axes):
    plt.plot(X[:, 0][y == 0], X[:, 1][y == 0], "bs")
    plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], "g^")
    plt.axis(axes)
    plt.grid(True, which="both")
    plt.xlabel(r"$x_1$", fontsize=20)
    plt.ylabel(r"$x_2$", fontsize=20, rotation=0)


def plot_predictions(clf, axes):
    x0s = np.linspace(axes[0], axes[1], 100)
    x1s = np.linspace(axes[2], axes[3], 100)
    x0, x1 = np.meshgrid(x0s, x1s)
    X = np.c_[x0.ravel(), x1.ravel()]
    y_pred = clf.predict(X).reshape(x0.shape)
    y_decision = clf.decision_function(X).reshape(x0.shape)
    plt.contourf(x0, x1, y_pred, cmap=plt.cm.brg, alpha=0.2)
    plt.contourf(x0, x1, y_decision, cmap=plt.cm.brg, alpha=0.1)


plot_predictions(polynomial_svm_clf, [-1.5, 2.5, -1, 1.5])
plot_datasets(X, y, [-1.5, 2.5, -1, 1.5])

save_fig("moons_polynomial_svc_plot")
plt.show()

# Polynomial Kernel

from sklearn.svm import SVC

poly_kernel_svm_clf = make_pipeline(
    StandardScaler(), SVC(kernel="poly", degree=3, coef0=1, C=5)
)

poly_kernel_svm_clf.fit(X, y)

poly100_kernel_svm_clf = make_pipeline(
    StandardScaler(), SVC(kernel="poly", degree=10, coef0=100, C=5)
)

poly100_kernel_svm_clf.fit(X, y)

plt.sca(axes[0])
plot_predictions(poly_kernel_svm_clf, [-1.5, 2.45, -1, 1.5])
plot_dataset(X, y, [-1.5, 2.4, -1, 1.5])
plt.title("degree=3, coef0=1, C=5")

plt.sca(axes[1])
plot_predictions(poly100_kernel_svm_clf, [-1.5, 2.45, -1, 1.5])
plot_dataset(X, y, [-1.5, 2.4, -1, 1.5])
plt.title("degree=10, coef0=100, C=5")
plt.ylabel("")

save_fig("moons_kernelized_polynomial_svc_plot")
plt.show()


# Similarity Features
def gaussian_rbf(X, landmark, gamma):
    return np.exp(-gamma * np.linalg.norm(X - landmark, axis=1) ** 2)


gamma = 0.3
x1s = np.linspace(-4.5, 4.5, 200).reshape(-1, 1)
x2s = gaussian_rbf(x1s, -2, gamma)
x3s = gaussian_rbf(x1s, 1, gamma)

XK = np.c_[gaussian_rbf(X1D, -2, gamma), gaussian_rbf(X1D, 1, gamma)]
yK = np.array([0, 0, 1, 1, 1, 1, 1, 0, 0])

plt.figure(figsize=(10.5, 4))

plt.subplot(121)
plt.grid(True)
plt.axhline(y=0, color="k")
plt.scatter(x=[-2, 1], y=[0, 0], s=150, alpha=0.5, c="red")
plt.plot(X1D[:, 0][yk == 0], np.zeros(4), "bs")
plt.plot(X1D[:, 0][yk == 1], np.zeros(5), "g^")
plt.plot(x1s, x2s, "g--")
plt.plot(x1s, x3s, "b:")
plt.gca().get_yaxis().set_ticks([0, 0.25, 0.5, 0.75, 1])
plt.xlabel("$x_1$")
plt.ylabel("Similarity")
plt.annotate(
    r"$\mathbf{x}$",
    xy=(X1D[3, 0], 0),
    xytext=(-0.5, 0.20),
    ha="center",
    arrowprops=dict(facecolor="black", shrink=0.1),
    fontsize=16,
)
plt.text(-2, 0.9, "$x_2$", ha="center", fontsize=15)
plt.text(1, 0.9, "$x_3$", ha="center", fontsize=15)
plt.axis([-4.5, 4.5, -0.1, 1.1])

plt.subplot(122)
plt.grid(True)
plt.axhline(y=0, color="k")
plt.axvline(x=0, color="k")
plt.plot(XK[:, 0][yk == 0], XK[:, 1][yk == 0], "bs")
plt.plot(XK[:, 0][yk == 1], XK[:, 1][yk == 1], "g^")
plt.xlabel("$x_2$")
plt.ylabel("$x_3$  ", rotation=0)
plt.annotate(
    r"$\phi\left(\mathbf{x}\right)$",
    xy=(XK[3, 0], XK[3, 1]),
    xytext=(0.65, 0.50),
    ha="center",
    arrowprops=dict(facecolor="black", shrink=0.1),
    fontsize=16,
)
plt.plot([-0.1, 1.1], [0.57, -0.1], "r--", linewidth=3)
plt.axis([-0.1, 1.1, -0.1, 1.1])

plt.subplots_adjust(right=1)

save_fig("kernel_method_plot")
plt.show()

# Gaussian RBF kernel

rbf_kernel_svm_clf = make_pipeline(StandardScaler, SVC(kernel="rbf", gamma=5, C=0.001))

rbf_kernel_svm_clf.fit(X, y)

from sklearn.svm import SVC

gamma1, gamma2 = 0.1, 5

C1, C2 = 0.001, 1000

hyperparams = (gamma1, C1), (gamma1, C2), (gamma2, C1), (gamma2, C2)

svm_clfs = []
for gamma, C in hyperparams:
    rbf_kernel_svm_clf = make_pipeline(
        StandardScaler(), SVC(kernel="rbf", gamma=gamma, C=C)
    )
    rbf_kernel_svm_clf.fit(X, y)
    svm_clfs.append(rbf_kernel_svm_clf)

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10.5, 7), sharex=True, sharey=True)

for i, svm_clf in enumerate(svm_clf):
    plt.sca(axes[i // 2, i % 2])
    plot_predictions(svm_clf, [-1.5, 2.45, -1, 1.5])
    plot_dataset(X, y, [-1.5, 2.45, -1, 1.5])
    gamma, C = hyperparams[i]
    plt.title(r"$\gamma = {}, C = {}$".format(gamma, C), fontsize=16)
    if i in (0, 1):
        plt.xlabel("")
    if i in (1, 3):
        plt.ylabel("")

save_fig("moons_rbf_svc_plot")
plt.show()

# SVM Regression

from sklearn.svm import LinearSVC, LinearSVR

np.random.seed(42)
X = 2 * np.random.rand(50, 1)
y = 4 + 3 * X[:, 0] + np.random.randn(50)

svm_reg = make_pipeline(StandardScaler(), LinearSVR(epsilon=1.5))

svm_reg.fit(X, y)


def find_support_vectors(svm_reg, X, y):
    y_pred = svm_reg.predict(X)
    off_margin = np.abs(y - y_pred) >= svm_reg.epsilon
    return np.argwhere(off_margin)


def plot_svm_regression(svm_reg, X, y, axes):
    x1s = np.linspace(axes[0], axes[1], 100).reshape(100, 1)
    y_pred = svm_reg.predict(x1s)
    plt.plot(x1s, y_pred, "k-", linewidth=2, label=r"$\hat{y}$")
    plt.plot(x1s, y_pred + svm_reg.epsilon, "k--")
    plt.plot(x1s, y_pred - svm_reg.epsilon, "k--")
    plt.scatter(X[svm_reg.support_], y[svm_reg.support_], s=180, facecolors="#FFAAAA")
    plt.plot(X, y, "bo")
    plt.xlabel(r"$x_1$", fontsize=18)
    plt.legend(loc="upper left", fontsize=18)
    plt.axis(axes)


svm_reg2 = make_pipeline(StandardScaler(), LinearSVR(epsilon=1.2, random_state=42))
svm_reg2.fit(X, y)

svm_reg._support = find_support_vectors(svm_reg, X, y)
svm_reg2._support = find_support_vectors(svm_reg2, X, y)

eps_x1 = 1
eps_y_pred = svm_reg2.predict([[eps_x1]])

fig, axes = plt.subplots(ncols=2, figsize=(9, 4), sharey=True)
plt.sca(axes[0])
plot_svm_regression(svm_reg, X, y, [0, 2, 3, 11])
plt.title(f"epsilon={svm_reg[-1].epsilon}")
plt.ylabel("$y$", rotation=0)
plt.grid()
plt.sca(axes[1])
plot_svm_regression(svm_reg2, X, y, [0, 2, 3, 11])
plt.title(f"epsilon={svm_reg2[-1].epsilon}")
plt.annotate(
    "",
    xy=(eps_x1, eps_y_pred),
    xycoords="data",
    xytext=(eps_x1, eps_y_pred - svm_reg2[-1].epsilon),
    textcoords="data",
    arrowprops={"arrowstyle": "<->", "linewidth": 1.5},
)
plt.text(0.90, 5.4, r"$\epsilon$", fontsize=16)
plt.grid()
save_fig("svm_regression_plot")
plt.show()

np.random.seed(42)
X = 2 * np.random.rand(100, 1) - 1
y = 0.2 + 0.1 * X[:, 0] + 0.5 * X[:, 0] ** 2 + np.random.rand(50) / 100

svm_poly_reg = make_pipeline(
    StandardScaler(), SVR(kernel="poly", degree=2, C=100, epsilon=0.1)
)
svm_poly_reg.fit(X, y)

# extra code – this cell generates and saves Figure 5–11

svm_poly_reg2 = make_pipeline(StandardScaler(), SVR(kernel="poly", degree=2, C=100))
svm_poly_reg2.fit(X, y)

svm_poly_reg._support = find_support_vectors(svm_poly_reg, X, y)
svm_poly_reg2._support = find_support_vectors(svm_poly_reg2, X, y)

fig, axes = plt.subplots(ncols=2, figsize=(9, 4), sharey=True)
plt.sca(axes[0])
plot_svm_regression(svm_poly_reg, X, y, [-1, 1, 0, 1])
plt.title(
    f"degree={svm_poly_reg[-1].degree}, "
    f"C={svm_poly_reg[-1].C}, "
    f"epsilon={svm_poly_reg[-1].epsilon}"
)
plt.ylabel("$y$", rotation=0)
plt.grid()

plt.sca(axes[1])
plot_svm_regression(svm_poly_reg2, X, y, [-1, 1, 0, 1])
plt.title(
    f"degree={svm_poly_reg2[-1].degree}, "
    f"C={svm_poly_reg2[-1].C}, "
    f"epsilon={svm_poly_reg2[-1].epsilon}"
)
plt.grid()
save_fig("svm_with_polynomial_kernel_plot")
plt.show()

# Under the hood

import matplotlib.patches as patches


def plot_2D_decision_function(w, b, ylabel=True, x1_lim=[-3, 3]):
    x1 = np.linspace(x1_lim[0], x1_lim[1], 200)
    y = w * x1 + b
    m = 1 / w
    plt.plot(x1, y, "k-", linewidth=2)
    plt.plot(x1_lim, [1, 1], "k--", linewidth=2)
    plt.plot(x1_lim, [-1, -1], "k--", linewidth=2)
    plt.axhline(y=0, color="k")
    plt.axvline(x=0, color="k")
    plt.plot([m, m], [0, 1], "k--", linewidth=2)
    plt.plot([-m, -m], [0, -1], "k--", linewidth=2)
    plt.fill_between(x1, y, 3, alpha=0.5)
    plt.fill_between(x1, y, -3, alpha=0.5)
    plt.axis(x1_lim + [-2, 2])
    plt.xlabel(r"$x_1$", fontsize=16)
    if ylabel:
        plt.ylabel(r"$w_1 x_1$  ", rotation=0, fontsize=16)
        plt.legend(loc="upper left")
        plt.text(m + 0.1, 0.5, "$m$", ha="center", fontsize=16)
    plt.title(f"$w_1 = {w}$", fontsize=16)

    plt.annotate(
        "",
        xy=(-half_margin, -1.6),
        xytext=(half_margin, -1.6),
        arrowprops={"ec": "k", "arrowstyle": "<->", "linewidth": 1.5},
    )
    plt.title(f"$w_1 = {w}$")


fig, axes = plt.subplots(ncols=2, figsize=(9, 3.2), sharey=True)
plt.sca(axes[0])
plot_2D_decision_function(1, 0)
plt.grid()
plt.sca(axes[1])
plot_2D_decision_function(0.5, 0, ylabel=False)
plt.grid()
save_fig("small_w_large_margin_plot")
plt.show()

# extra code – this cell generates and saves Figure 5–13

s = np.linspace(-2.5, 2.5, 200)
hinge_pos = np.where(1 - s < 0, 0, 1 - s)  # max(0, 1 - s)
hinge_neg = np.where(1 + s < 0, 0, 1 + s)  # max(0, 1 + s)

titles = (r"Hinge loss = $max(0, 1 - s\,t)$", "Squared Hinge loss")

fix, axs = plt.subplots(1, 2, sharey=True, figsize=(8.2, 3))

for ax, loss_pos, loss_neg, title in zip(
    axs, (hinge_pos, hinge_pos**2), (hinge_neg, hinge_neg**2), titles
):
    ax.plot(s, loss_pos, "g-", linewidth=2, zorder=10, label="$t=1$")
    ax.plot(s, loss_neg, "r--", linewidth=2, zorder=10, label="$t=-1$")
    ax.grid(True)
    ax.axhline(y=0, color="k")
    ax.axvline(x=0, color="k")
    ax.set_xlabel(r"$s = \mathbf{w}^\intercal \mathbf{x} + b$")
    ax.axis([-2.5, 2.5, -0.5, 2.5])
    ax.legend(loc="center right")
    ax.set_title(title)
    ax.set_yticks(np.arange(0, 2.5, 1))
    ax.set_aspect("equal")

save_fig("hinge_plot")
plt.show()

# extra material

# Linear SVM classifier implementation using Batch Gradient Descent

# Load the iris dataset

X = iris["data"][:, (2, 3)]  # petal length, petal width
y = (iris["target"] == 2).astype(np.float64).reshape(-1, 1)  # Iris-Virginica

from sklearn.base import BaseEstimator


class MyLinearSVC(BaseEstimator):
    def __init__(self, C=1, eta0=1, eta_d=10000, n_epochs=1000, random_state=None):
        self.C = C
        self.eta0 = eta0
        self.eta_d = eta_d
        self.n_epochs = n_epochs
        self.random_state = random_state

    def eta(self, epoch):
        return self.eta0 / (epoch + self.eta_d)

    def fit(self, X, y):
        # Random initialization
        if self.random_state:
            np.random.seed(self.random_state)
        w = np.random.randn(X.shape[1], 1)
        b = 0
        for epoch in range(self.n_epochs):
            for i in range(len(X)):
                if y[i] * (X[i] @ w + b) < 1:
                    gradient_w = -X[i].T * y[i]
                    gradient_b = -y[i]
                else:
                    gradient_w = 0
                    gradient_b = 0
                w = w - self.eta(epoch) * (w + self.C * gradient_w)
                b = b - self.eta(epoch) * self.C * gradient_b

        self.w_ = w
        self.b_ = b
        return self

    def decision_function(self, X):
        return X @ self.w_ + self.b_

    def predict(self, X):
        return (self.decision_function(X) >= 0).astype(np.float64)


C = 2
svm_clf = MyLinearSVC(C=C, eta0=10, eta_d=1000, n_epochs=60000, random_state=2)
svm_clf.fit(X, y.ravel())
svm_clf.predict(np.array([[5, 2], [4, 1]]))

plt.plot(range(svm_clf.n_epochs), svm_clf.Js)
plt.axis([0, svm_clf.n_epochs, 0, 100])
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.grid()
plt.show()

print(svm_clf.intercept_, svm_clf.coef_)

svm_clf2 = SVC(kernel="linear", C=C)
svm_clf2.fit(X, y.ravel())
print(svm_clf2.intercept_, svm_clf2.coef_)

yr = y.ravel()
fig, axes = plt.subplots(ncols=2, figsize=(11, 3.2), sharey=True)
plt.sca(axes[0])
plt.plot(X[:, 0][yr == 1], X[:, 1][yr == 1], "g^", label="Iris virginica")
plt.plot(X[:, 0][yr == 0], X[:, 1][yr == 0], "bs", label="Not Iris virginica")
plot_svc_decision_boundary(svm_clf, 4, 6)
plt.xlabel("Petal length")
plt.ylabel("Petal width")
plt.title("MyLinearSVC")
plt.axis([4, 6, 0.8, 2.8])
plt.legend(loc="upper left")
plt.grid()

plt.sca(axes[1])
plt.plot(X[:, 0][yr == 1], X[:, 1][yr == 1], "g^")
plt.plot(X[:, 0][yr == 0], X[:, 1][yr == 0], "bs")
plot_svc_decision_boundary(svm_clf2, 4, 6)
plt.xlabel("Petal length")
plt.title("SVC")
plt.axis([4, 6, 0.8, 2.8])
plt.grid()

plt.show()

# Exercise solutions

# 1. to 10.

# What is the fundamental idea behind Support Vector Machines?

# The fundamental idea behind Support Vector Machines is to fit the widest
# possible "street" between the classes. In other words, the goal is to have
# the largest possible margin between the decision boundary that separates
# the two classes and the training instances. When performing soft margin
# classification, the SVM searches for a compromise between perfectly
# separating the two classes and having the widest possible street (i.e.,
# a few instances may end up on the street). Another key idea is to use
# kernels when training on nonlinear datasets.

# What is a support vector?

# Once the SVM classifier is trained, the prediction phase is very similar
# to that of a linear classifier: it just computes the decision function
# $w^T \cdot x + b = w_1 x_1 + \dots + w_n x_n + b$. If the result is
# positive, the predicted class $\hat{y}$ is the positive class (1), or else
# it is the negative class (0); see Equation 5-2. Equation 5-2. Linear SVM
# classifier prediction $\hat{y} = \begin{cases} 0 & \text{if } w^T \cdot x + b
# < 0, \\ 1 & \text{if } w^T \cdot x + b \ge 0. \end{cases}$
# The decision boundary is the set of points where the decision function is
# equal to 0: it is the intersection of two planes, which is a straight line
# (represented by the thick solid line). The dashed lines represent the points
# where the decision function is equal to 1 or –1: they are parallel and at
# equal distance to the decision boundary, forming a margin around it. Training
# a linear SVM classifier means finding the value of $w$ and $b$ that make this
# margin as wide as possible while avoiding margin violations (hard margin) or
# limiting them (soft margin).

# Can an SVM classifier output a confidence score when it classifies an
# instance? What about a probability?

# An SVM classifier can output the distance between the test instance and the
# decision boundary, and you can use this as a confidence score. However, this
# score cannot be directly converted into an estimation of the class
# probability. If you set `probability=True` when creating an SVM in Scikit-Learn,
# then after training it will calibrate the probabilities using Logistic
# Regression on the SVM’s scores (trained by an additional five-fold
# cross-validation on the training data). This will add the `predict_proba()`
# and `predict_log_proba()` methods to the SVM.

# Should you scale the inputs before training an SVM?

# SVMs are sensitive to the feature scales, as you can see in Figure 5-8. On
# the left plot, the vertical scale is much larger than the horizontal scale,
# so the widest possible street is close to horizontal. After feature scaling
# (e.g., using Scikit-Learn’s `StandardScaler`), the decision boundary looks
# much better (on the right plot).

# How do you train an SVM classifier on an out-of-core dataset?

# The `LinearSVC` class is based on the `liblinear` library, which implements
# an optimized algorithm for linear SVMs. This algorithm does not support
# the kernel trick, but it scales almost linearly with the number of training
# instances and the number of features: its training time complexity is
# roughly $O(m \times n)$. The algorithm takes longer if you require a very
# high precision. This is controlled by the tolerance hyperparameter $\epsilon$
# (called `tol` in Scikit-Learn). In most classification tasks, the default
# tolerance is fine.

# The `SVC` class is based on the `libsvm` library, which implements an
# algorithm that supports the kernel trick. The training time complexity is
# usually between $O(m^2 \times n)$ and $O(m^3 \times n)$. Unfortunately, this
# means that it gets dreadfully slow when the number of training instances
# gets large (e.g., hundreds of thousands of instances). This algorithm is
# perfect for complex but small or medium training sets. However, it scales
# well with the number of features, especially with sparse features (i.e.,
# when each instance has few nonzero features). In this case, the algorithm
# scales roughly with the average number of nonzero features per instance.
# Table 5-1 compares Scikit-Learn’s SVM classification classes.

# Table 5-1. SVM classifiers comparison
# Class | Time complexity | Out-of-core support | Scaling required | Kernel trick
# --- | --- | --- | --- | ---
# `LinearSVC` | $O(m \times n)$ | No | Yes | No
# `SGDClassifier` | $O(m \times n)$ | Yes | Yes | No
# `SVC` | $O(m^2 \times n)$ to $O(m^3 \times n)$ | No | Yes | Yes

# Train a LinearSVC on a linearly separable dataset. Then train an SVC and
# a SGDClassifier on the same dataset. See if you can get them to produce
# roughly the same model.

# First, let’s load the iris dataset and split it into a training set and a
# test set:

X = iris["data"][:, (2, 3)]  # petal length, petal width
yr = (iris["target"] == 2).astype(np.float64)  # Iris virginica

# Now let’s train the models:

from sklearn.svm import SVC, LinearSVC

svm_clf1 = SVC(kernel="linear", C=1e9)
svm_clf1.fit(X, yr)

# The `LinearSVC` class regularizes the bias term, so you should center the
# training set first by subtracting its mean. This is automatic if you scale
# the data using the `StandardScaler`. Moreover, make sure you set the
# `loss` hyperparameter to `"hinge"`, as it is not the default value. Finally,
# for better performance you should set the `dual` hyperparameter to `False`,
# unless there are more features than training instances (we will discuss
# duality later in the chapter).

from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(
    loss="hinge", alpha=1 / (len(X) * 1e9), max_iter=1000, tol=1e-3, random_state=42
)

sgd_clf.fit(X, yr)

# Let’s plot the decision boundaries of these three models:

# Compute the slope and bias of each decision boundary

w1 = -svm_clf1.coef_[0, 0] / svm_clf1.coef_[0, 1]

b1 = -svm_clf1.intercept_[0] / svm_clf1.coef_[0, 1]

w2 = -sgd_clf.coef_[0, 0] / sgd_clf.coef_[0, 1]

b2 = -sgd_clf.intercept_[0] / sgd_clf.coef_[0, 1]

# Transform the decision boundary lines back to the original scale

line1 = scaler.inverse_transform([[-10, -10 * w1 + b1], [10, 10 * w1 + b1]])

line2 = scaler.inverse_transform([[-10, -10 * w2 + b2], [10, 10 * w2 + b2]])

# Plot all three decision boundaries

plt.figure(figsize=(11, 4))

plt.plot(line1[:, 0], line1[:, 1], "k:", label="LinearSVC")

plt.plot(line2[:, 0], line2[:, 1], "b--", linewidth=2, label="SGDClassifier")

plt.plot(X[:, 0][yr == 1], X[:, 1][yr == 1], "g^")

plt.plot(X[:, 0][yr == 0], X[:, 1][yr == 0], "bs")

plt.xlabel("Petal length", fontsize=14)

plt.ylabel("Petal width", fontsize=14)

plt.legend(loc="upper center", fontsize=14)

plt.axis([0, 5.5, 0, 2])

plt.show()

# As you can see, they all find a good solution, but `LinearSVC` does not
# output a probability vector; instead, it just relies on the decision
# function, so it does not provide `predict_proba()` method. Moreover, it
# does not scale as well as `SGDClassifier` or `SVC` when the number of
# features grows large (such as when you are using sparse features).

# If you want to have `predict_proba()` (to get the class probabilities) or
# `decision_function()` (to get the distance to the decision boundary) for a
# `LinearSVC` classifier, you need to set the `loss` hyperparameter to
# `"hinge"`, as shown in the left plot of Figure 5-9. This will make the
# `LinearSVC` class use the same loss function as the `SVC` class. Similarly,
# you can set `SGDClassifier(loss="hinge")`. Finally, if you want better
# performance, you should set the `dual` hyperparameter to `False`, unless
# there are more features than training instances (we will discuss duality
# later in the chapter).

# Now let’s try the SVC class. It is based on the libsvm library, which
# implements an algorithm that supports the kernel trick. Let’s test it on
# the iris dataset:

svm_clf2 = SVC(kernel="poly", degree=3, coef0=1, C=5)

svm_clf2.fit(X, yr)

# The hyperparameter `coef0` controls how much the model is influenced by
# high-degree polynomials versus low-degree polynomials.

# Finally, let’s try a model based on the Gaussian RBF kernel. It is often
# good at classification tasks:
# $$\phi_\gamma(\mathbf{x}, \ell) = \exp(-\gamma \|\mathbf{x} - \ell\|^2)$$
# It is a bell-shaped function varying from 0 (very far away from the landmark)
# to 1 (at the landmark). Now we are ready to train the model:

svm_clf3 = SVC(kernel="rbf", gamma=5, C=0.001)

svm_clf3.fit(X, yr)

# Figure 5-9 shows the decision boundaries and support vectors of these three

# models. Notice that adding more training instances within the margin does

# not affect the decision boundary at all: it is fully determined (or

# "supported") by the instances located on the edge of the margin. These

# instances are called the support vectors (they are circled in Figure 5-9).

# Notice that they do not necessarily be located on the edge of the street

# since the margin is soft. If we strictly impose that all instances be off

# the street and on the right side, this is called hard margin classification.

# There are two main issues with hard margin classification. First, it only

# works if the data is linearly separable, and second it is quite sensitive to

# outliers. Figure 5-10 shows the iris dataset with just one additional

# outlier: on the left, it is impossible to find a hard margin, and on the

# right the decision boundary ends up very different from the one we saw in

# Figure 5-9 without the outlier, and it will probably not generalize as well:

# ![Figure 5-10](images/05_10.png)

# To avoid these issues it is preferable to use a more flexible model. The

# objective is to find a good balance between keeping the street as large as

# possible and limiting the margin violations (i.e., instances that end up in

# the middle of the street or even on the wrong side). This is called soft

# margin classification.


# ![Figure 5-9](images/05_09.png)

# If your `SVM` model is overfitting, you can try regularizing it by reducing

# `C`. Conversely, if it is underfitting, you can try increasing `C` (similar

# to the `C` hyperparameter of the `LinearSVC` class). The following

# `SVC` model is regularized because `C` is set to a small value:

# **Warning**: the `gamma` hyperparameter acts like a regularization

# hyperparameter: if your model is overfitting, you should reduce it. Conversely,

# if it is underfitting, you should increase it (similar to the `alpha`

# hyperparameter of linear regression models).

# Other kernels exist but are used much more rarely. For example, some kernels

# are specialized for specific data structures. String kernels are sometimes

# used when classifying text documents or DNA sequences (e.g., using the

# string subsequence kernel or kernels based on the Levenshtein distance).

# ![Figure 5-11](images/05_11.png)

# **Tip**: With so many kernels to choose from, how can you decide which one to

# use? As a rule of thumb, you should always try the linear kernel first

# (remember that `LinearSVC` is much faster than `SVC(kernel="linear")`), especially

# if the training set is very large or if it has plenty of features. If the

# training set is not too large, you should try the Gaussian RBF kernel as

# well; it works well in most cases. Then if you have spare time and computing

# power, you can also experiment with a few other kernels using crossvalidation

# and grid search, especially if there are kernels specialized for

# your training set’s data structure.

# Train an SVM classifier on the MNIST dataset. Since SVM classifiers are
# binary classifiers, you will need to use one-versus-all to classify all 10
# digits. You may want to tune the hyperparameters using small validation
# sets to speed up the process. What accuracy can you reach?

# First, let's load the data and split it into a training set and a test set.

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

mnist = fetch_openml("mnist_784", version=1, cache=True)

X_train = mnist["data"][:60000]
X_test = mnist["data"][60000:]
y_train = mnist["target"][:60000]
y_test = mnist["target"][60000:]

# The dataset is actually already split into a training set (the first 60,000
# images) and a test set (the last 10,000 images):

# The training set is already shuffled for us, which is good as this

# guarantees that all cross-validation folds will be similar (you don’t want

# one fold to be missing some digits). Moreover, some learning algorithms are

# sensitive to the order of the training instances, and they perform poorly if

# they get many similar instances in a row. Shuffling the dataset ensures that

# this won’t happen. However, this does mean that a simple `cross_val_score()`

# (or `cross_val_predict()`) call would not be appropriate, since it would

# assume that the data is not shuffled.

# Let’s create a function that will shuffle the dataset, then create a

# `StratifiedKFold` and use it to perform stratified sampling (with 3 folds)

# to create a training set and a test set:

from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

skfolds = StratifiedKFold(n_splits=3, random_state=42)


def shuffleDataSet(X, y):
    shuffle_index = np.random.permutation(60000)
    return X[shuffle_index], y[shuffle_index]


def getTrainingAndTestSet(X, y):
    for train_index, test_index in skfolds.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
    return X_train, X_test, y_train, y_test


# Let’s start simple, with a linear `SVC`. It will automatically use the
# one-versus-all (also called one-versus-the-rest, `OvR`) strategy, so there is
# nothing special we need to do. This time we do not want to use the `StandardScaler`
# to scale the data:

from sklearn.svm import SVC

X_train, X_test, y_train, y_test = getTrainingAndTestSet(X_train, y_train)

svm_clf = SVC(kernel="linear", random_state=42)

svm_clf.fit(X_train, y_train)

# Let’s make predictions on the training set and measure the accuracy:

from sklearn.metrics import accuracy_score

y_pred = svm_clf.predict(X_train)

accuracy_score(y_train, y_pred)

# Okay, 86% accuracy on the training set is not that great, but let’s see how

# it does on the test set:

y_pred = svm_clf.predict(X_test)

accuracy_score(y_test, y_pred)

# Okay, this is not as good as the accuracy we got with Random Forests, but

# still not too bad. Let’s see if we can improve this score by scaling the

# inputs. We will use a `StandardScaler`. Note that we could also have used a

# `MinMaxScaler` instead. The result is not exactly the same since the

# `StandardScaler` centers the data around zero while the `MinMaxScaler` scales

# the data to a given range of values (e.g., 0-1), but it should be close

# enough for our purposes.

# Let’s scale the inputs to mean 0 and unit variance:

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))

X_test_scaled = scaler.fit_transform(X_test.astype(np.float64))

# Now let’s train an `SVC` on the scaled training set:

svm_clf.fit(X_train_scaled, y_train)

# Let’s make predictions on the scaled training set and measure the accuracy:

y_pred = svm_clf.predict(X_train_scaled)

accuracy_score(y_train, y_pred)

# Great, this time the accuracy is much better. Now let’s see how it does on
# the test set:

y_pred = svm_clf.predict(X_test_scaled)

accuracy_score(y_test, y_pred)

# That’s a significant improvement, well done!

# Let’s try the `SGDClassifier` class, training it with default hyperparameters
# (i.e., `loss="hinge"`, `alpha=1/(m*C)`). It should give us similar results to
# the `LinearSVC` class:

from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(random_state=42)

sgd_clf.fit(X_train_scaled, y_train)

y_pred = sgd_clf.predict(X_test_scaled)

accuracy_score(y_test, y_pred)

# The `SGDClassifier` converges after 3 epochs. So it’s not as good as `LinearSVC`,
# but `SGDClassifier` is useful for handling huge datasets that do not fit in
# memory (out-of-core training), or for online learning tasks.

# Let’s see if we can do better by searching for better hyperparameters using
# `GridSearchCV`. We will do this on the small training set for the sake of
# keeping the example short:

from sklearn.model_selection import GridSearchCV

param_grid = [
    {"C": [0.1, 1, 10, 100, 1000], "kernel": ["linear"]},
    {"C": [0.1, 1, 10, 100, 1000], "gamma": [0.001, 0.01, 0.1, 1], "kernel": ["rbf"]},
]

svm_clf = SVC()

grid_search = GridSearchCV(
    svm_clf, param_grid, cv=5, scoring="accuracy", return_train_score=True
)

grid_search.fit(X_train_scaled[:1000], y_train[:1000])

# Train an SVM regressor on the California housing dataset:

from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing()

X = housing["data"]
y = housing["target"]

X_train, X_test, y_train, y_test = getTrainingAndTestSet(X, y)

from sklearn.svm import LinearSVR

svm_reg = LinearSVR(random_state=42)

svm_reg.fit(X_train, y_train)

# Let’s measure this model’s RMSE on the whole training set:

from sklearn.metrics import mean_squared_error

y_pred = svm_reg.predict(X_train)

mse = mean_squared_error(y_train, y_pred)

np.sqrt(mse)

# Okay, that’s clearly not a great score: most districts’ `median_housing_values`

# range between $120,000 and $265,000, so a typical prediction error of $68,628

# is not very satisfying. Let’s see if we can do better with an RBF Kernel. We

# will use `RandomizedSearchCV` to find the appropriate hyperparameter values

# for `C` and `gamma`:

from sklearn.svm import SVR

from sklearn.model_selection import RandomizedSearchCV

param_distributions = {"C": reciprocal(20, 200000), "gamma": expon(scale=1.0)}

svm_reg = SVR()

rnd_search_cv = RandomizedSearchCV(
    svm_reg, param_distributions, n_iter=10, verbose=2, cv=5, random_state=42
)

rnd_search_cv.fit(X_train_scaled, y_train)

# The best model achieves the following score (evaluated using 5-fold cross

# validation):

rnd_search_cv.best_estimator_

# Let’s select this model and evaluate it on the test set:

y_pred = rnd_search_cv.best_estimator_.predict(X_train_scaled)

mse = mean_squared_error(y_train, y_pred)

np.sqrt(mse)

# Let’s see how the best model performs on the test set:

y_pred = rnd_search_cv.best_estimator_.predict(X_test_scaled)

mse = mean_squared_error(y_test, y_pred)

np.sqrt(mse)

# Well, that’s a lot better than the linear model. Let’s check the best

# hyperparameters found:

rnd_search_cv.best_estimator_

# The `RandomizedSearchCV` class will use cross-validation to evaluate

# each combination of hyperparameter values you specify. If you set

# `cv=5`, then it will use 5-fold cross-validation (so it will evaluate

# each combination of hyperparameter values 5 times, since we have one

# hyperparameter, this means that it will train and evaluate 5 × 5 = 25

# times). In other words, that’s equivalent to 25 rounds of training

# (on the full training set) and evaluating (on the full validation set).

# In the end, the `RandomizedSearchCV` instance will use the best

# hyperparameter values found, train a final model on these, and keep

# this model as the best estimator.

# The advantage of `RandomizedSearchCV` over `GridSearchCV` is that

# you can control the number of iterations (`n_iter`) and have more

# control over the computing budget you want to allocate to hyperparameter

# search.

# When using small datasets, it is often preferable to use `GridSearchCV`,

# but for larger datasets with a large number of hyperparameters, `RandomizedSearchCV`

# is often preferable. You can also use `RandomizedSearchCV` with `GridSearchCV`

# together. Simply treat `RandomizedSearchCV` as a single hyperparameter search

# method, and if you have the computing power, you can use it to search for the

# best hyperparameters values (as well as for the best number of iterations).

# The following code searches for the best combination of hyperparameter values

# for an `SVR` with an RBF kernel. It searches among 3 values for `C` (100, 1000,

# and 10000), 3 values for `gamma` (0.01, 0.1, and scale), and 3 values for `epsilon`

# (0.1, 1.0, and 10.0), so in total `3 × 3 × 3 = 27` combinations of hyperparameter

# values. It will use 5-fold cross-validation, so it will actually perform `27 × 5 = 135`

# rounds of training and evaluate on the training set. In other words, it will take

# quite a long time (may be a few hours depending on the hardware).

from scipy.stats import reciprocal, expon

from sklearn.model_selection import RandomizedSearchCV

param_distributions = {
    "kernel": ["rbf"],
    "C": reciprocal(20, 200000),
    "gamma": expon(scale=1.0),
}

svm_reg = SVR()

rnd_search_cv = RandomizedSearchCV(
    svm_reg, param_distributions, n_iter=10, verbose=2, cv=5, random_state=42
)

rnd_search_cv.fit(X_train_scaled[:1000], y_train[:1000])

# The best model achieves the following score on the training set:

rnd_search_cv.best_estimator_

# And the best hyperparameters are:

rnd_search_cv.best_estimator_

# The best model achieves the following score on the test set:

rnd_search_cv.best_estimator_

# The `RandomizedSearchCV` class also supports `Scikit-Learn`’s `Pipeline` class.

# Let’s create a pipeline containing a standard scaler, followed by a `SVR`:

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from scipy.stats import reciprocal, expon

from sklearn.model_selection import RandomizedSearchCV

param_distributions = {
    "kernel": ["rbf"],
    "C": reciprocal(20, 200000),
    "gamma": expon(scale=1.0),
}

svm_reg = SVR()

rnd_search_cv = RandomizedSearchCV(
    svm_reg, param_distributions, n_iter=10, verbose=2, cv=5, random_state=42
)

rnd_search_cv.fit(X_train_scaled[:1000], y_train[:1000])

# The best model achieves the following score on the training set:

rnd_search_cv.best_estimator_

# And the best hyperparameters are:

rnd_search_cv.best_estimator_

# The best model achieves the following score on the test set:

rnd_search_cv.best_estimator_

# The `RandomizedSearchCV` class also supports `Scikit-Learn`’s `Pipeline` class.

# Let’s create a pipeline containing a standard scaler, followed by a `SVR`:

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from scipy.stats import reciprocal, expon

from sklearn.model_selection import RandomizedSearchCV

param_distributions = {
    "kernel": ["rbf"],
    "C": reciprocal(20, 200000),
    "gamma": expon(scale=1.0),
}

svm_reg = SVR()

rnd_search_cv = RandomizedSearchCV(
    svm_reg, param_distributions, n_iter=10, verbose=2, cv=5, random_state=42
)

rnd_search_cv.fit(X_train_scaled[:1000], y_train[:1000])

# The best model achieves the following score on the training set:

rnd_search_cv.best_estimator_

# And the best hyperparameters are:

rnd_search_cv.best_estimator_

# The best model achieves the following score on the test set:

rnd_search_cv.best_estimator_

# The `RandomizedSearchCV` class also supports `Scikit-Learn`’s `Pipeline` class.
