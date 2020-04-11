from utils import load_mnist
from sklearn.manifold import Isomap


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA, NMF
from sklearn.feature_selection import SelectKBest, chi2


x_train, _, y_train, _ = load_mnist(permutation=True, train_size=100, test_size=100)

model = Isomap(n_components=2)
proj = model.fit_transform(x_train)
plt.scatter(proj[:, 0], proj[:, 1], c=y_train, cmap=plt.cm.get_cmap('jet', 10))
plt.colorbar(ticks=range(10))
plt.clim(-0.5, 9.5)

