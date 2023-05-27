# setup

import sys

assert sys.version_info >= (3, 7)

import numpy as np
import pandas as pd

import sklearn

assert sklearn.__version__ >= "0.20"

import matplotlib.pyplot as plt

root_dir = "F:/GitHubRepos/hands-on-machine-learning/my_work/"

plt.rc("font", size=14)
plt.rc("axes", labelsize=14, titlesize=14)
plt.rc("legend", fontsize=14)
plt.rc("xtick", labelsize=10)
plt.rc("ytick", labelsize=10)

from pathlib import Path

IMAGES_PATH = Path() / "images" / "classification"
IMAGES_PATH.mkdir(parents=True, exist_ok=True)


def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = IMAGES_PATH / f"{fig_id}.{fig_extension}"
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


# MNIST

from sklearn.datasets import fetch_openml

mnist = fetch_openml("mnist_784", version=1, as_frame=False)

print(mnist.DESCR)

mnist.keys()

X, y = mnist.data, mnist.target

X

X.shape

y

y.shape

import matplotlib.pyplot as plt


def plot_digit(image_data):
    image = image_data.reshape(28, 28)
    plt.imshow(image, cmap="binary")
    plt.axis("off")


some_digit = X[0]
plot_digit(some_digit)
# append root_dir to save_fig function
image_dir = root_dir + "images/chapter3/"
save_fig(image_dir + "some_digit_plot.svg")
plt.show()

y[0]

# extra code – this cell generates and saves Figure 3–2
plt.figure(figsize=(9, 9))
for idx, image_data in enumerate(X[:100]):
    plt.subplot(10, 10, idx + 1)
    plot_digit(image_data)
plt.subplots_adjust(wspace=0, hspace=0)
save_fig(image_dir + "more_digits_plot.svg", tight_layout=False)
plt.show()

X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

# Binary classifier

y_train_5 = y_train == "5"
y_test_5 = y_test == "5"

from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)

sgd_clf.predict([some_digit])

##Performance measures##

# Measuring accuracy using cross-validation

from sklearn.model_selection import cross_val_score

cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")

# Implementing cross-validation

from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

skfolds = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

for train_index, test_index in skfolds.split(X_train, y_train_5):
    clone_clf = clone(sgd_clf)
    X_train_folds = X_train[train_index]
    y_train_folds = y_train_5[train_index]
    X_test_fold = X_train[test_index]
    y_test_fold = y_train_5[test_index]

    clone_clf.fit(X_train_folds, y_train_folds)
    y_pred = clone_clf.predict(X_test_fold)
    n_correct = sum(y_pred == y_test_fold)
    print(n_correct / len(y_pred))

# extra code – this cell generates and saves Figure 3–3

from sklearn.utils import DummyClassifier

dmy_clf = DummyClassifier(strategy="stratified")
dmy_clf.fit(X_train, y_train_5)
print(any(dmy_clf.predict(X_train)))

cross_val_score(dmy_clf, X_train, y_train_5, cv=3, scoring="accuracy")

# Confusion matrix

from sklearn.utils import cross_val_predict

y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_train_5, y_train_pred)

cm

y_train_perfect_predictions = y_train_5
confusion_matrix(y_train_5, y_train_perfect_predictions)

# Precision and recall

from sklearn.metrics import precision_score, recall_score

precision_score(y_train_5, y_train_pred)

recall_score(y_train_5, y_train_pred)

cm[1, 1] / (cm[0, 1] + cm[1, 1])

recall_score(y_train_5, y_train_pred)

cm[1, 1] / (cm[1, 0] + cm[1, 1])

from sklearn.metrics import f1_score

f1_score(y_train_5, y_train_pred)

# extra code – this cell generates and saves Figure 3–4

cm[1, 1] / (cm[1, 1] + (cm[cm[1, 1], cm[0, 1]]) / 2)

# Precision/recall trade-off

y_scores = sgd_clf.decision_function([some_digit])

y_scores

threshold = 0

y_some_digit_pred = y_scores > threshold

y_scores > 0

threshold = 8000

y_some_digit_pred = y_scores > threshold

y_some_digit_pred

y_scores = cross_val_predict(
    sgd_clf, X_train, y_train_5, cv=3, method="decision_function"
)

from sklearn.metrics import precision_recall_curve

precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)


def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)
    plt.legend(loc="center right", fontsize=16)
    plt.xlabel("Threshold", fontsize=16)
    plt.grid(True)
    plt.axis([-50000, 50000, 0, 1])


plt.figure(figsize=(8, 4))
plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.plot([7816, 7816], [0.0, 0.9], "r:")
plt.plot([-50000, 7816], [0.9, 0.9], "r:")
plt.plot([-50000, 7816], [0.4368, 0.4368], "r:")
plt.plot([7816], [0.9], "ro")
plt.plot([7816], [0.4368], "ro")
save_fig(image_dir + "precision_recall_vs_threshold_plot.svg")
plt.show()

(y_train_pred == (y_scores > 0)).all()

y_train_pred_90 = y_scores > 7816

precision_score(y_train_5, y_train_pred_90)

recall_score(y_train_5, y_train_pred_90)

import matplotlib.patches as patches


def plot_precision_vs_recall(precisions, recalls):
    plt.plot(recalls, precisions, "b-", linewidth=2)
    plt.xlabel("Recall", fontsize=16)
    plt.ylabel("Precision", fontsize=16)
    plt.axis([0, 1, 0, 1])
    plt.grid(True)


plt.figure(figsize=(8, 6))
plot_precision_vs_recall(precisions, recalls)
plt.plot([0.4368, 0.4368], [0.0, 0.9], "r:")
plt.plot([0.0, 0.4368], [0.9, 0.9], "r:")
plt.plot([0.4368], [0.9], "ro")
save_fig(image_dir + "precision_vs_recall_plot.svg")
plt.show()

idx_for_90_precision = np.argmax(precisions >= 0.90)
threshold_90_precision = thresholds[idx_for_90_precision]
threshold_90_precision

y_train_pred_90 = y_scores >= threshold_90_precision

precision_score(y_train_5, y_train_pred_90)

recall_at_90_precision = recalls[idx_for_90_precision]
recall_at_90_precision

recall_score(y_train_5, y_train_pred_90)

# The ROC curve

from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)

idx_for_threshold_at_90 = (thresholds <= threshold_90_precision).argmin()
tpr_90, fpr_90 = tpr[idx_for_threshold_at_90], fpr[idx_for_threshold_at_90]


def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], "k--")
    plt.axis([0, 1, 0, 1])
    plt.xlabel("False Positive Rate (Fall-Out)", fontsize=16)
    plt.ylabel("True Positive Rate (Recall)", fontsize=16)
    plt.grid(True)


plt.figure(figsize=(8, 6))
plot_roc_curve(fpr, tpr)
plt.plot([fpr_90], [tpr_90], "ro")
plt.plot([fpr_90, fpr_90], [0.0, tpr_90], "r:")
plt.plot([0.0, fpr_90], [tpr_90, tpr_90], "r:")
save_fig(image_dir + "roc_curve_plot.svg")
plt.show()
save_fig(image_dir + "roc_curve_plot.svg")

from sklearn.metrics import roc_auc_score

roc_auc_score(y_train_5, y_scores)

from sklearn.ensemble import RandomForestClassifier

forest_clf = RandomForestClassifier(random_state=42)

y_probas_forest = cross_val_predict(
    forest_clf, X_train, y_train_5, cv=3, method="predict_proba"
)

y_probas_forest[:2]

idx_50_to_60 = (y_probas_forest[:, 1] >= 0.5) & (y_probas_forest[:, 1] < 0.6)
print(f"{y_train_5[idx_50_to_60].sum() / idx_50_to_60.sum():.1%}")

y_scores_forest = y_probas_forest[:, 1]  # score = proba of positive class

precision_forest, recalls_forest, thresholds_forest = precision_recall_curve(
    y_train_5, y_scores_forest
)

plt.figure(figsize=(8, 6))

plt.plot(
    thresholds_forest, precision_forest[:-1], "b--", label="Precision", linewidth=2
)
plt.plot(thresholds_forest, recalls_forest[:-1], "g-", label="Recall", linewidth=2)
plt.legend(loc="center right", fontsize=16)

plt.xlabel("Threshold", fontsize=16)
plt.grid(True)
plt.axis([-0.5, 1.5, 0, 1])
plt.ylabel("Score", fontsize=16)
save_fig(image_dir + "precision_recall_vs_threshold_plot_forest.svg")

plt.show()

y_train_pred_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3)

f1_score(y_train_5, y_train_pred_forest)

roc_auc_score(y_train_5, y_scores_forest)
precision_score(y_train_5, y_train_pred_forest)
recall_score(y_train_5, y_scores_forest)

# Multiclass classification

from sklearn.svm import SVC

svm_clf = SVC(gamma="auto", random_state=42)
svm_clf.fit(X_train[:1000], y_train[:1000])  # y_train, not y_train_5

svm_clf.predict([some_digit])

some_digit_scores = svm_clf.decision_function([some_digit])
some_digit_scores.round(2)

class_id = some_digit_scores.argmax()
class_id

svm_clf.classes_

svm_clf.classes_[class_id]

svm_clf.decision_function_shape = "ovo"
some_digits_score_ovo = svm_clf.decision_function([some_digit])
some_digits_score_ovo.round(2)

from sklearn.multiclass import OneVsRestClassifier

ovr_clf = OneVsRestClassifier(SVC(gamma="auto", random_state=42))
ovr_clf.fit(X_train[:1000], y_train[:1000])

ovr_clf.predict([some_digit])

len(ovr_clf.estimators_)

sdg_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train)
sgd_clf.predict([some_digit])

sdg_clf.decision_function([some_digit])

cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring="accuracy")

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
cross_val_score(sgd_clf, X_train_scaled, y_train, cv=3, scoring="accuracy")

y_train_pred = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=3)
conf_mx = confusion_matrix(y_train, y_train_pred)
conf_mx

plt.matshow(conf_mx, cmap=plt.cm.gray)
save_fig(image_dir + "confusion_matrix_plot", tight_layout=False)
plt.show()

row_sums = conf_mx.sum(axis=1, keepdims=True)
norm_conf_mx = conf_mx / row_sums

np.fill_diagonal(norm_conf_mx, 0)
plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
save_fig(image_dir + "confusion_matrix_errors_plot", tight_layout=False)
plt.show()

cl_a, cl_b = 3, 5
X_aa = X_train[(y_train == cl_a) & (y_train_pred == cl_a)]
X_ab = X_train[(y_train == cl_a) & (y_train_pred == cl_b)]
X_ba = X_train[(y_train == cl_b) & (y_train_pred == cl_a)]
X_bb = X_train[(y_train == cl_b) & (y_train_pred == cl_b)]

plt.figure(figsize=(8, 8))
plt.subplot(221)
plot_digits(X_aa[:25], images_per_row=5)
plt.subplot(222)
plot_digits(X_ab[:25], images_per_row=5)
plt.subplot(223)
plot_digits(X_ba[:25], images_per_row=5)
plt.subplot(224)
plot_digits(X_bb[:25], images_per_row=5)
save_fig(image_dir + "error_analysis_digits_plot")

# error analysis

from sklearn.metrics import ConfusionMatrixDisplay

y_train_pred = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=3)
conf_mx = confusion_matrix(y_train, y_train_pred)
plt.rc("font", size=16)
ConfusionMatrixDisplay.from_predictions(y_train, y_train_pred, cmap=plt.cm.gray).plot()
save_fig(image_dir + "confusion_matrix_plot", tight_layout=False)
plt.show()

plt.rct("font", size=16)
ConfusionMatrixDisplay.from_predictions(y_train, y_train_pred, cmap=plt.cm.gray).plot()
save_fig(image_dir + "confusion_matrix_errors_plot", tight_layout=False)
plt.show()

sample_weight = y_train_pred != y_train
plt.rc("font", size=16)
ConfusionMatrixDisplay.from_predictions(
    y_train, y_train_pred, cmap=plt.cm.gray, sample_weight=sample_weight
).plot()
save_fig(image_dir + "confusion_matrix_errors_plot", tight_layout=False)
plt.show()

# extra code – this cell generates and saves Figure 3–9
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(9, 4))
plt.rc("font", size=9)
ConfusionMatrixDisplay.from_predictions(y_train, y_train_pred, ax=axs[0])
axs[0].set_title("Confusion matrix")
plt.rc("font", size=10)
ConfusionMatrixDisplay.from_predictions(
    y_train, y_train_pred, ax=axs[1], normalize="true", values_format=".0%"
)
axs[1].set_title("CM normalized by row")
save_fig("confusion_matrix_plot_1")
plt.show()

# extra code – this cell generates and saves Figure 3–10
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(9, 4))
plt.rc("font", size=10)
ConfusionMatrixDisplay.from_predictions(
    y_train,
    y_train_pred,
    ax=axs[0],
    sample_weight=sample_weight,
    normalize="true",
    values_format=".0%",
)
axs[0].set_title("Errors normalized by row")
ConfusionMatrixDisplay.from_predictions(
    y_train,
    y_train_pred,
    ax=axs[1],
    sample_weight=sample_weight,
    normalize="pred",
    values_format=".0%",
)
axs[1].set_title("Errors normalized by column")
save_fig("confusion_matrix_plot_2")
plt.show()
plt.rc("font", size=14)  # make fonts great again

cl_a, cl_b = 3, 5
X_aa = X_train[(y_train == cl_a) & (y_train_pred == cl_a)]
X_ab = X_train[(y_train == cl_a) & (y_train_pred == cl_b)]
X_ba = X_train[(y_train == cl_b) & (y_train_pred == cl_a)]
X_bb = X_train[(y_train == cl_b) & (y_train_pred == cl_b)]

plt.figure(figsize=(8, 8))
plt.subplot(221)
plot_digits(X_aa[:25], images_per_row=5)
plt.subplot(222)
plot_digits(X_ab[:25], images_per_row=5)
plt.subplot(223)
plot_digits(X_ba[:25], images_per_row=5)
plt.subplot(224)
plot_digits(X_bb[:25], images_per_row=5)

save_fig(image_dir + "error_analysis_digits_plot")

# extra code – this cell generates and saves Figure 3–11
size = 5
pad = 0.2
plt.figure(figsize=(size, size))
for images, (label_col, label_row) in [
    (X_ba, (0, 0)),
    (X_bb, (1, 0)),
    (X_aa, (0, 1)),
    (X_ab, (1, 1)),
]:
    for idx, image_data in enumerate(images[: size * size]):
        x = idx % size + label_col * (size + pad)
        y = idx // size + label_row * (size + pad)
        plt.imshow(
            image_data.reshape(28, 28), cmap="binary", extent=(x, x + 1, y, y + 1)
        )
plt.xticks([size / 2, size + pad + size / 2], [str(cl_a), str(cl_b)])
plt.yticks([size / 2, size + pad + size / 2], [str(cl_b), str(cl_a)])
plt.plot([size + pad / 2, size + pad / 2], [0, 2 * size + pad], "k:")
plt.plot([0, 2 * size + pad], [size + pad / 2, size + pad / 2], "k:")
plt.axis([0, 2 * size + pad, 0, 2 * size + pad])
plt.xlabel("Predicted label")
plt.ylabel("True label")
save_fig("error_analysis_digits_plot")
plt.show()

# Multilabel classification

import numpy as np
from sklearn.neighbors import KNeighborsClassifier

y_train_large = y_train >= 7
y_train_odd = y_train % 2 == 1
y_multilabel = np.c_[y_train_large, y_train_odd]

knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_multilabel)

knn_clf.predict([some_digit])

y_train_knn_pred = cross_val_predict(knn_clf, X_train, y_multilabel, cv=3)
f1_score(y_multilabel, y_train_knn_pred, average="macro")

f1_score(y_multilabel, y_train_knn_pred, average="weighted")

from sklearn.multioutput import MultiOutputClassifier

chain_clf = ClassifierChain(SVC(), cv=3, random_state=42)
chain_clf.fit(X_train[:2000], y_mutilabel[:2000])

chain_clf.predict([some_digit])

# Multioutput classification

np.random.seed(42)
noise = np.random.randint(0, 100, (len(X_train), 784))
X_train_mod = X_train + noise
noise = np.random.randint(0, 100, (len(X_test), 784))
X_test_mod = X_test + noise
y_train_mod = X_train
y_test_mod = X_test

some_index = 0
plt.subplot(121)
plot_digit(X_test_mod[some_index])
plt.subplot(122)
plot_digit(y_test_mod[some_index])
save_fig(image_dir + "noisy_digit_example_plot")
plt.show()

knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train_mod, y_train_mod)
clean_digit = knn_clf.predict([X_test_mod[some_index]])
plot_digit(clean_digit)
save_fig(image_dir + "cleaned_digit_example_plot")

# Exercise: _Try to build a classifier for the MNIST dataset that achieves over 97% accuracy on the test set. Hint: the `KNeighborsClassifier` works quite well for this task; you just need to find good hyperparameter values (try a grid search on the `weights` and `n_neighbors` hyperparameters)._

# get the basline k-nearest neighbour classifier as a reference

from sklearn.model_selection import GridSearchCV

knn_clf = KNeighborsClassifier()
param_grid = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
grid_search = GridSearchCV(knn_clf, param_grid, cv=5, verbose=3, n_jobs=-1)
grid_search.fit(X_train, y_train)

grid_search.best_params_

grid_search.best_score_

from sklearn.metrics import accuracy_score

y_pred = grid_search.predict(X_test)
accuracy_score(y_test, y_pred)

# Data augmentation

from scipy.ndimage.interpolation import shift

knn_clf = KNeighborsClassifier(**grid_search.best_params_)
knn_clf.fit(X_train, y_train)
knn_clf.score(X_test, y_test)

dx = 1
dy = 0

shifted_images = [shift(image, [0, dx, dy]) for image in X_train]

X_train_augmented = np.array(X_train + shifted_images)
y_train_augmented = np.array(y_train + [label for label in y_train for _ in range(4)])

shuffle_idx = np.random.permutation(len(X_train_augmented))
X_train_augmented = X_train_augmented[shuffle_idx]
y_train_augmented = y_train_augmented[shuffle_idx]

knn_clf.fit(X_train_augmented, y_train_augmented)
y_pred = knn_clf.predict(X_test)
accuracy_score(y_test, y_pred)

# Exercise: Write a function with the MNIST classifier that can shift the image in any direction

# Tackle the Titanic dataset

import os
import pandas as pd

TITANIC_PATH = os.path.join("datasets", "titanic")

from sklearn.base import BaseEstimator, TransformerMixin

train_data = pd.read_csv(TITANIC_PATH + "/train.csv")


class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.attribute_names].values


from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

num_pipeline = Pipeline(
    [
        ("select_numeric", DataFrameSelector(["Age", "SibSp", "Parch", "Fare"])),
        ("imputer", SimpleImputer(strategy="median")),
    ]
)

num_pipeline.fit_transform(train_data)

from sklearn.preprocessing import OneHotEncoder

# cat_pipeline = Pipeline([knn_clf = KNeighborsClassifier(**grid_search.best_params_)])

from sklearn.pipeline import FeatureUnion

preprocess_pipeline = FeatureUnion(transformer_list=["num_pipeline", "cat_pipeline"])

X_train = preprocess_pipeline.fit_transform(train_data)
X_train

y_train = train_data["Survived"]

from sklearn.svm import SVC

svm_clf = SVC(gamma="auto")
svm_clf.fit(X_train, y_train)

from sklearn.model_selection import cross_val_score

svm_scores = cross_val_score(svm_clf, X_train, y_train, cv=10)

svm_scores.mean()

from sklearn.ensemble import RandomForestClassifier

forest_clf = RandomForestClassifier(n_estimators=100, random_state=42)

forest_scores = cross_val_score(forest_clf, X_train, y_train, cv=10)

forest_scores.mean()

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 4))

plt.plot([1] * 10, svm_scores, ".")

plt.plot([2] * 10, forest_scores, ".")

plt.boxplot([svm_scores, forest_scores], labels=("SVM", "Random Forest"))

plt.ylabel("Accuracy", fontsize=14)

save_fig(image_dir + "boxplot_comparison_plot")

plt.show()

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV

svm_clf = Pipeline([("scaler", StandardScaler()), ("svm_clf", SVC(gamma="auto"))])

param_grid = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

grid_search = GridSearchCV(svm_clf, param_grid, cv=5, verbose=3, n_jobs=-1)

grid_search.fit(X_train, y_train)

grid_search.best_params_

grid_search.best_score_

from sklearn.model_selection import RandomizedSearchCV

from scipy.stats import reciprocal, uniform

param_distributions = {"gamma": reciprocal(0.001, 0.1), "C": uniform(1, 10)}

rnd_search_cv = RandomizedSearchCV(
    svm_clf, param_distributions, n_iter=10, verbose=2, cv=3, n_jobs=-1
)

rnd_search_cv.fit(X_train, y_train)

rnd_search_cv.best_score_

rnd_search_cv.best_estimator_

rnd_search_cv.best_estimator_.fit(X_train, y_train)

X_test = preprocess_pipeline.transform(test_data)

y_pred = rnd_search_cv.best_estimator_.predict(X_test)

# Data augmentation

from scipy.ndimage import shift


def shift_image(image, dx, dy):
    image = image.reshape((28, 28))
    shifted_image = shift(image, [dy, dx], cval=0, mode="constant")
    return shifted_image.reshape([-1])


image = X_train[1000]
shifted_image_down = shift_image(image, 0, 5)
shifted_image_left = shift_image(image, -5, 0)

plt.figure(figsize=(12, 3))
plt.subplot(131)
plt.title("Original", fontsize=14)
plt.imshow(image.reshape(28, 28), interpolation="nearest", cmap="Greys")
plt.subplot(132)
plt.title("Shifted down", fontsize=14)
plt.imshow(shifted_image_down.reshape(28, 28), interpolation="nearest", cmap="Greys")
plt.subplot(133)
plt.title("Shifted left", fontsize=14)
plt.imshow(shifted_image_left.reshape(28, 28), interpolation="nearest", cmap="Greys")
plt.subplot(134)
plt.title("Shifted down", fontsize=14)
plt.imshow(shifted_image_down.reshape(28, 28), interpolation="nearest", cmap="Greys")
plt.subplot(135)
plt.title("Shifted left", fontsize=14)
plt.imshow(shifted_image_left.reshape(28, 28), interpolation="nearest", cmap="Greys")
plt.show()

X_train_augmented = [image for image in X_train]
y_train_augmented = [label for label in y_train]

for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
    for image, label in zip(X_train, y_train):
        X_train_augmented.append(shift_image(image, dx, dy))
        y_train_augmented.append(label)

X_train_augmented = np.array(X_train_augmented)
y_train_augmented = np.array(y_train_augmented)

shuffle_idx = np.random.permutation(len(X_train_augmented))
X_train_augmented = X_train_augmented[shuffle_idx]
y_train_augmented = y_train_augmented[shuffle_idx]

knn_clf = KNeighborsClassifier(**grid_search.best_params_)
knn_clf.fit(X_train_augmented, y_train_augmented)

y_pred = knn_clf.predict(X_test)

augmented_accuracy = knn_clf.score(X_test, y_test)
augmented_accuracy = np.mean(y_pred == y_test)
augmented_accuracy = np.max(y_pred == y_test)

error_rate_change = (1 - augmented_accuracy) / (1 - accuracy) - 1
print(f"error rate change: {error_rate_change:.4f}")

# Tackle the Titanic dataset

import os
from pathlib import Path
import pandas as pd
import numpy as np
import tarfile
import urlib.request as ur

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
TITANIC_PATH = os.path.join("datasets", "titanic")


def load_titanic_data():
    csv_path = os.path.join(TITANIC_PATH, "train.csv")
    return pd.read_csv(csv_path)


titanic = load_titanic_data()

train_data, test_data = load_titanic_data()

train_data.head()

train_data = train_data.set_index("PassengerId")
test_data = test_data.set_index("PassengerId")

train_data.info()

train_data.describe()

train_data[train_data["Sex"] == "Female"]["Age"].median()

train_data.describe()

train_data["Survived"].value_counts()

train_data["Pclass"].value_counts()

train_data["Sex"].value_counts()

train_data["Embarked"].value_counts()

train_data["SibSp"].value_counts()

train_data["Parch"].value_counts()

train_data["Cabin"].value_counts()

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

num_pipeline = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", SimpleImputer(strategy="median")),
    ]
)

from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

cat_pipeline = Pipeline(
    [("imputer", SimpleImputer(strategy="most_frequent")), ("cat", OneHotEncoder())]
)

from sklearn.compose import ColumnTransformer

num_attribs = ["Age", "SibSp", "Parch", "Fare"]
cat_attribs = ["Age", "SibSp", "Parch", "Fare"]

preprocess_pipeline = ColumnTransformer(
    [("num", num_pipeline, num_attribs), ("cat", cat_pipeline, cat_attribs)]
)

X_train = preprocess_pipeline.fit_transform(X_train)
X_train

y_train = train_data["Survived"]

forest_clf = RandomForestClassifier(n_estimators=100, random_state=42)
forest_clf.fit(X_train, y_train)

X_test = preprocess_pipeline.transform(test_data)
y_pred = forest_clf.predict(X_test)

forest_scores = cross_val_score(forest_clf, y_pred, cv=10)
forest_scores.mean()

from sklearn.svm import SVC

svm_clf = SVC(gamma="auto")
svm_scores = cross_val_score(svm_clf, y_pred, cv=10)
svm_scores.mean()

plt.figure(figsize=(8, 4))
plt.plot([1] * 10, forest_scores, ".")
plt.plot([2] * 10, svm_scores, ".")
plt.boxplot([forest_scores, svm_scores], labels=("Random Forest", "SVM"))
plt.ylabel("Accuracy", fontsize=14)
plt.show()
plt.save_fig(os.path.join(os.path.dirname()))

train_data["AgeBucket"] = train_data["Age"] // 15 * 15
train_data[["AgeBucket", "Survived"]].groupby(["AgeBucket"]).mean()

train_data["RelativesOnboard"] = train_data["SibSp"] + train_data["Parch"]
train_data[["RelativesOnboard", "Survived"]].groupby(["RelativesOnboard"]).mean()

train_data = train_data.drop(["AgeBucket", "RelativesOnboard"], axis=1)
train_data.head()

train_data["Deck"] = train_data["Cabin"].str[0]
train_data[["Deck", "Survived"]].groupby(["Deck"]).mean()

train_data = train_data.drop(["Cabin", "Ticket"], axis=1)
train_data.head()

train_data["Title"] = train_data["Name"].str.extract(" ([A-Za-z]+)\.", expand=False)
train_data["Title"].value_counts()

train_data["Title"] = train_data["Title"].replace(
    [
        "Lady",
        "Countess",
        "Capt",
        "Col",
        "Don",
        "Dr",
        "Major",
        "Rev",
        "Sir",
        "Jonkheer",
        "Dona",
    ],
    "Rare",
)

train_data["Title"] = train_data["Title"].replace("Mlle", "Miss")
train_data["Title"] = train_data["Title"].replace("Ms", "Miss")
train_data["Title"] = train_data["Title"].replace("Mme", "Mrs")

train_data[["Title", "Survived"]].groupby(["Title"]).mean()

train_data = train_data.drop(["Name"], axis=1)
train_data.head()

# Spam classifier

import os
import tarfile


def fetch_spam_data():
    DOWNLOAD_ROOT = "https://spamassassin.apache.org/old/publiccorpus/"
    HAM_URL = DOWNLOAD_ROOT + "20030228_easy_ham.tar.bz2"
    SPAM_URL = DOWNLOAD_ROOT + "20030228_spam.tar.bz2"
    SPAM_PATH = os.path.join("datasets", "spam")

    if not os.path.isdir(SPAM_PATH):
        os.makedirs(SPAM_PATH)
    for filename, url in (("ham.tar.bz2", HAM_URL), ("spam.tar.bz2", SPAM_URL)):
        path = os.path.join(SPAM_PATH, filename)
        if not os.path.isfile(path):
            urllib.request.urlretrieve(url, path)
        tar_bz2_file = tarfile.open(path)
        tar_bz2_file.extractall(path=SPAM_PATH)
        tar_bz2_file.close()
    return SPAM_


ham_dir, spam_dir = fetch_spam_data()

ham_filenames = [f for f in sorted(os.listdir(ham_dir)) if len(f) > 20]
spam_filenames = [f for f in sorted(os.listdir(spam_dir)) if len(f) > 20]

len(ham_filenames)

len(spam_filenames)

import email
import email.policy


def load_email(is_spam, filename, spam_path=SPAM_PATH):
    directory = "spam" if is_spam else "easy_ham"
    with open(os.path.join(spam_path, directory, filename), "rb") as f:
        return email.parser.BytesParser(policy=email.policy.default).parse(f)


ham_emails = [load_email(is_spam=False, filename=name) for name in ham_filenames]
spam_emails = [load_email(is_spam=True, filename=name) for name in spam_filenames]

ham_emails[1].get_content().strip()

print(ham_emails[1].get_content().strip())


def get_email_structure(email):
    if isinstance(email, str):
        return email
    payload = email.get_payload()
    if isinstance(payload, list):
        return "multipart({})".format(
            ", ".join([get_email_structure(sub_email) for sub_email in payload])
        )
    else:
        return email.get_content_type()


from collections import Counter


def structures_counter(emails):
    structures = Counter()
    for email in emails:
        structure = get_email_structure(email)
        structures[structure] += 1
    return structures


structures_counter(ham_emails).most_common()

structures_counter(spam_emails).most_common()

for header, value in spam_emails[0].items():
    print(header, ":", value)

spam_emails[0]["Subject"]

import numpy as np
from sklearn.model_selection import train_test_split

X = np.array(ham_emails + spam_emails, dtype=object)
y = np.array([0] * len(ham_emails) + [1] * len(spam_emails))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

import re
from html import unescape


def html_to_plain_text(html):
    text = re.sub("<head.*?>.*?</head>", "", html, flags=re.M | re.S | re.I)
    text = re.sub("<a\s.*?>", " HYPERLINK ", text, flags=re.M | re.S | re.I)
    text = re.sub("<.*?>", "", text, flags=re.M | re.S)
    text = re.sub(r"(\s*\n)+", "\n", text, flags=re.M | re.S)
    return unescape(text)


html_spam_emails = [
    email
    for email in X_train[y_train == 1]
    if get_email_structure(email) == "text/html"
]
sample_html_spam = html_spam_emails[7]
print(sample_html_spam.get_content().strip()[:1000], "...")
print(html_to_plain_text(sample_html_spam.get_content())[:1000], "...")

import nltk

stemmer = nltk.PorterStemmer()

for word in (
    "Computations",
    "Computation",
    "Computing",
    "Computed",
    "Compute",
    "Compulsive",
):
    print(word, "=>", stemmer.stem(word))

import urlextract

url_extractor = urlextract.URLExtract()

print(
    url_extractor.find_urls(
        "Will it detect github.com and https://youtu.be/7Pq-S557XQU?t=3m32s"
    )
)

from sklearn.base import BaseEstimator, TransformerMixin


class EmailToWordCounterTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        strip_headers=True,
        lower_case=True,
        remove_punctuation=True,
        replace_urls=True,
        replace_numbers=True,
        stemming=True,
    ):
        self.strip_headers = strip_headers
        self.lower_case = lower_case
        self.remove_punctuation = remove_punctuation
        self.replace_urls = replace_urls
        self.replace_numbers = replace_numbers
        self.stemming = stemming

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_transformed = []
        for email in X:
            text = email_to_text(email)
            if self.lower_case:
                text = text.lower()
            if self.replace_urls and url_extractor is not None:
                urls = list(set(url_extractor.find_urls(text)))
                urls.sort(key=lambda url: len(url), reverse=True)
                for url in urls:
                    text = text.replace(url, " URL ")
            if self.replace_numbers:
                text = re.sub(r"\d+(?:\.\d*(?:[eE]\d+))?", "NUMBER", text)
            if self.remove_punctuation:
                text = re.sub(r"\W+", " ", text, flags=re.M)
            word_counts = Counter(text.split())
            if self.stemming and stemmer is not None:
                stemmed_word_counts = Counter()
                for word, count in word_counts.items():
                    stemmed_word = stemmer.stem(word)
                    stemmed_word_counts[stemmed_word] += count
                word_counts = stemmed_word_counts
            X_transformed.append(word_counts)
        return np.array(X_transformed)


X_few = X_train[:3]
X_few_wordcounts = EmailToWordCounterTransformer().fit_transform(X_few)
X_few_wordcounts

from scipy.sparse import csr_matrix


class WordCounterToVectorTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, vocabulary_size=1000):
        self.vocabulary_size = vocabulary_size

    def fit(self, X, y=None):
        total_count = Counter()
        for word_count in X:
            for word, count in word_count.items():
                total_count[word] += min(count, 10)

        most_common = total_count.most_common()[: self.vocabulary_size]
        self.most_common_ = most_common
        self.vocabulary_ = {
            word: index + 1 for index, (word, count) in enumerate(most_common)
        }
        return self

    def transform(self, X, y=None):
        rows = []
        cols = []
        data = []
        for row, word_count in enumerate(X):
            for word, count in word_count.items():
                rows.append(row)
                cols.append(self.vocabulary_.get(word, 0))
                data.append(count)

        return csr_matrix(
            (data, (rows, cols)), shape=(len(X), self.vocabulary_size + 1)
        )


vocab_transformer = WordCounterToVectorTransformer(vocabulary_size=10)
X_few_vectors = vocab_transformer.fit_transform(X_few_wordcounts)
X_few_vectors

X_few_vectors.toarray()

vocab_transformer.vocabulary_

from sklearn.pipeline import Pipeline

preprocess_pipeline = Pipeline(
    [
        ("email_to_wordcount", EmailToWordCounterTransformer()),
        ("wordcount_to_vector", WordCounterToVectorTransformer()),
    ]
)

X_train_transformed = preprocess_pipeline.fit_transform(X_train)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

log_clf = LogisticRegression(solver="liblinear", random_state=42)
score = cross_val_score(log_clf, X_train_transformed, y_train, cv=3, verbose=3)
score.mean()

from sklearn.metrics import precision_score, recall_score

X_test_transformed = preprocess_pipeline.transform(X_test)

log_clf = LogisticRegression(solver="liblinear", random_state=42)

log_clf.fit(X_train_transformed, y_train)

y_pred = log_clf.predict(X_test_transformed)

precision_score(y_test, y_pred)

recall_score(y_test, y_pred)

from sklearn.metrics import f1_score

f1_score(y_test, y_pred)

from sklearn.metrics import precision_recall_curve

y_scores = cross_val_predict(
    log_clf, X_train_transformed, y_train, cv=3, method="decision_function", verbose=3
)

precisions, recalls, thresholds = precision_recall_curve(y_train, y_scores)


def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)
    plt.xlabel("Threshold", fontsize=16)
    plt.legend(loc="upper left", fontsize=16)
    plt.ylim([0, 1])


plt.figure(figsize=(8, 4))
plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.xlim([-5, 5])
plt.show()
plt.save_fig("precision_recall_vs_threshold.png")

(y_train_pred == (y_scores > 0)).all()
