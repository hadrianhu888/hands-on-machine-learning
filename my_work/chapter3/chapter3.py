import sys

assert sys.version_info >= (3, 7)

import numpy as np
import pandas as pd

import sklearn

assert sklearn.__version__ >= "0.20"

import matplotlib.pyplot as plt

root_dir = "F:/GitHubRepos/hands-on-machine-learning/my_work/"

plt.rc('font', size=14)
plt.rc('axes', labelsize=14, titlesize=14)
plt.rc('legend', fontsize=14)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)

from pathlib import Path

IMAGES_PATH = Path() / "images" / "classification"
IMAGES_PATH.mkdir(parents=True, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = IMAGES_PATH / f"{fig_id}.{fig_extension}"
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)
    
#MNIST 
    
from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784', version=1, as_frame=False)

print(mnist.DESCR)

mnist.keys()

X,y = mnist.data, mnist.target

X

X.shape

y

y.shape

import matplotlib.pyplot as plt 

def plot_digit(image_data):
    image = image_data.reshape(28,28)
    plt.imshow(image,cmap='binary')
    plt.axis("off")
    
some_digit = X[0]
plot_digit(some_digit)
#append root_dir to save_fig function
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

X_train,X_test, y_train,y_test = X[:60000], X[60000:], y[:60000], y[60000:]

# Binary classifier

y_train_S = (y_train == '5')
y_test_S = (y_test == '5')

from sklearn.linear_model import SGDClassifier

sdg_clf = SGDClassifier(random_state=42)
sdg_clf.fit(X_train,y_train_S)

sdg_clf.predict([some_digit])

# Performance measures

# Cross validation

from sklearn.model_selection import cross_val_score

cross_val_score(sdg_clf,X_train,y_train_S,cv=3,scoring="accuracy")

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.base import clone

skfolds = StratifiedGroupKFold(n_splits=3, random_state=42)

for train_index, test_index in skfolds.split(X_train,y_train_S):
    clone_clf = clone(sdg_clf)
    X_train_folds = X_train[train_index]
    y_train_folds = y_train_S[train_index]
    x_test_fold = X_train[test_index]
    y_test_fold = y_train_S[test_index]
    
    clone_clf.fit(X_train_folds,y_train_folds)
    y_pred = clone_clf.predict(x_test_fold)
    n_correct = sum(y_pred == y_test_fold)
    print(n_correct / len(y_pred))
    
from sklearn.dummy import DummyClassifier

dummy_clf = DummyClassifier(strategy="most_frequent")
dummy_clf.fit(X_train,y_train_S)
print(any(dummy_clf.predict([some_digit])))

cross_val_score(dummy_clf,X_train,y_train_S,cv=3,scoring="accuracy")

# Confusion matrix

from sklearn.model_selection import cross_val_predict
y_train_pred = cross_val_predict(sdg_clf,X_train,y_train_S,cv=3)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_train_S,y_train_pred)
cm

y_train_perfect_predictions = y_train_S
confusion_matrix(y_train_S, y_train_perfect_predictions)

# Precision and recall

y_scores = cross_val_predict(sdg_clf,X_train,y_train_S,cv=3,method="decision_function")
y_scores

threshold = 0

y_some_digit_pred = (y_scores > threshold)

y_some_digit_pred

y_scores > 0

y_scores = cross_val_predict(sdg_clf,X_train,y_train_S,cv=3,method="decision_function")

from sklearn.metrics import precision_score, recall_score

precisions, recalls, thresholds = precision_recall_curve(y_train_S,y_scores)

plt.figure(figsize=(8, 4))  # extra code – it's not needed, just formatting
plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)
plt.vlines(threshold, 0, 1.0, "k", "dotted", label="threshold")

# extra code – this section just beautifies and saves Figure 3–5
idx = (thresholds >= threshold).argmax()  # first index ≥ threshold
plt.plot(thresholds[idx], precisions[idx], "bo")
plt.plot(thresholds[idx], recalls[idx], "go")
plt.axis([-50000, 50000, 0, 1])
plt.grid()
plt.xlabel("Threshold")
plt.legend(loc="center right")
save_fig(image_dir + "precision_recall_vs_threshold_plot.svg", tight_layout=False)

plt.show()

idx_for_90_precision = np.argmax(precisions >= 0.90)
threshold_for_90_precision = thresholds[idx_for_90_precision]
threshold_for_90_precision

y_train_pred_90 = (y_scores >= threshold_for_90_precision)

precision_score(y_train_S,y_train_pred_90)

recall_at_90_precision = recalls[idx_for_90_precision]
recall_at_90_precision

# ROC curve

from sklearn.metrics import roc_curve 
fpr,tpr,thresholds=  roc_curve(y_train_S,y_scores)

idx_for_threshold_at_90 = (thresholds <= threshold_for_90_precision).argmax()
tpr_90, fpr_90 = tpr[idx_for_threshold_at_90], fpr[idx_for_threshold_at_90]

plt.figure(figsize=(6, 5))  # extra code – not needed, just formatting
plt.plot(fpr, tpr, linewidth=2, label="ROC curve")
plt.plot([0, 1], [0, 1], 'k:', label="Random classifier's ROC curve")
plt.plot([fpr_90], [tpr_90], "ko", label="Threshold for 90% precision")

# extra code – just beautifies and saves Figure 3–7
plt.gca().add_patch(patches.FancyArrowPatch(
    (0.20, 0.89), (0.07, 0.70),
    connectionstyle="arc3,rad=.4",
    arrowstyle="Simple, tail_width=1.5, head_width=8, head_length=10",
    color="#444444"))
plt.text(0.12, 0.71, "Higher\nthreshold", color="#333333")
plt.xlabel('False Positive Rate (Fall-Out)')
plt.ylabel('True Positive Rate (Recall)')
plt.grid()
plt.axis([0, 1, 0, 1])
plt.legend(loc="lower right", fontsize=13)
save_fig(image_dir + "roc_curve_plot.svg", tight_layout=False)

plt.show()

from sklearn.metrics import roc_auc_score

roc_auc_score(y_train_S,y_scores)

from sklearn.ensemble import RandomForestClassifier

forest_clf = RandomForestClassifier(random_state=42)

y_probas_forest = cross_val_predict(forest_clf,X_train,y_train_S,cv=3,method="predict_proba")

y_probas_forest[:5]

idx_50_to_90 = (y_scores >= threshold_for_90_precision).argmax()

print(f"{(y_scores >= threshold_for_90_precision).sum()} / {len(y_scores)} = {100 * (y_scores >= threshold_for_90_precision).sum() / len(y_scores)}")

y_scores_forest = y_probas_forest[:, 1]  # score = proba of positive class

precisions_forest, recalls_forest, thresholds_forest = precision_recall_curve(y_train_S,y_scores_forest)


plt.figure(figsize=(6, 5))  # extra code – not needed, just formatting

plt.plot(recalls_forest, precisions_forest, "b-", linewidth=2,label="Random Forest")
plt.plot(recalls, precisions, "--", linewidth=2, label="SGD")

# extra code – just beautifies and saves Figure 3–8
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.axis([0, 1, 0, 1])
plt.grid()
plt.legend(loc="lower left")
save_fig(image_dir + "pr_curve_comparison_plot.svg", tight_layout=False)

plt.show()

y_train_pred_forest = y_probas_forest[:, 1] > 0.5
f1_score(y_train_S,y_train_pred_forest)

roc_auc_score(y_train_S,y_scores_forest)

precision_score(y_train_S,y_train_pred_forest)

recall_score(y_train_S,y_train_pred_forest)

# Multiclass classification

