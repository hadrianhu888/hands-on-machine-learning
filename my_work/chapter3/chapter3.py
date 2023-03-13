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



