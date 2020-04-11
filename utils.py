from sklearn.datasets import fetch_openml
from sklearn.datasets import make_swiss_roll
from sklearn.datasets import make_s_curve
from sklearn.utils import check_random_state
from sklearn.model_selection import train_test_split
import numpy as np
import os 

def load_mnist(permutation = True, train_size = 5000, test_size = 10000):
    """
        Load dataset from https://openml.org/d/554
        MNIST dataset consisting of 60'000 training set and 10'000 test set. 
    """
    print("* Loading MNIST Dataset")
    x, y = fetch_openml('mnist_784', version=1, return_X_y = True, data_home=os.path.join(os.getcwd(), 'data'))
    print("* Loaded MNIST Dataset")
    if permutation:
        random_state = check_random_state(0) ## Can change the seed
        p = random_state.permutation(x.shape[0])
        x = x[p]
        y = y[p]
        x = x.reshape((x.shape[0], -1))
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train_size, test_size=test_size)
    return x_train, x_test, y_train, y_test

def load_swiss_roll(n_points = 2000, deviation = 0.1):
    x, color = make_swiss_roll(n_samples = n_points, noise = deviation)
    return x, color

def load_s_curve(n_points = 2000, deviation = 0.1):
    x, color = make_s_curve(n_samples = n_points, noise = deviation)
    return x, color

