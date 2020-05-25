"""
Module utilities with useful functions
"""
import os 
from sklearn.datasets import fetch_openml
from sklearn.datasets import make_swiss_roll
from sklearn.datasets import make_s_curve
from sklearn.datasets import load_digits
import numpy as np
from sklearn.utils import check_random_state


def load_mnist_sk(n_class=10):
    X, y = load_digits(n_class=n_class, return_X_y=True)
    return X, y

def load_mnist(permutation=True, n_class=10):
    """
        Load dataset from https://openml.org/d/554
        MNIST dataset consisting of 60'000 training set and 10'000 test set. 
    """
    print("* Loading MNIST Dataset")
    x, y = fetch_openml(
        'mnist_784',
         version=1,
         return_X_y=True,
         data_home=os.path.join(os.getcwd(), 'data')
    )

    if permutation:
        random_state = check_random_state(0) ## Can change the seed
        p = random_state.permutation(x.shape[0])
        x = x[p]
        y = y[p]
        x = x.reshape((x.shape[0], -1))
    y = y.astype(np.int)
    if n_class == 10:
        return x, y
    else:
        idx = np.argwhere(y < n_class).tolist()
        x = x[idx]
        x = x.reshape((x.shape[0], -1))
        return x, y[idx]

def load_mnist_fashion(permutation=True, n_class=10):
    """
        Load dataset from https://www.openml.org/d/40996
        MNIST Fashion dataset consisting of 60'000 training set and 10'000 test set. 
    """
    print("* Loading MNIST FashionDataset")
    x, y = fetch_openml(
        'Fashion-MNIST',
        version=1,
        return_X_y=True,
        data_home=os.path.join(os.getcwd(), 'data')
    )

    if permutation:
        random_state = check_random_state(0) ## Can change the seed
        p = random_state.permutation(x.shape[0])
        x = x[p]
        y = y[p]
        x = x.reshape((x.shape[0], -1))
    y = y.astype(np.int)
    if n_class == 10:
        return x, y
    else:
        idx = np.argwhere(y < n_class).tolist()
        x = x[idx]
        x = x.reshape((x.shape[0], -1))
        return x, y[idx]

def load_swiss_roll(n_points = 2000, deviation = 0.1):
    """
    Load a swiss roll dataset
    """
    x, color = make_swiss_roll(n_samples=n_points, noise=deviation)
    return x, color

def load_s_curve(n_points = 2000, deviation = 0.1):
    """
    Load a s curve dataset
    """
    x, color = make_s_curve(n_samples=n_points, noise=deviation)
    return x, color

def load_s_curve_hole(n_points=2000, deviation = 0.1):
    """
    Load a s curve dataset but with a hole in the middle
    """
    x, color = make_s_curve(n_samples=n_points, noise=deviation)



    return x, color

def load_data(d_type, n_samples = 2000, dev = 0, n_class=10):
    """
    Load data handler
    """
    if d_type == 's_curve':
        print('* Load S Curve')
        x, label = load_s_curve(n_samples, dev)
        return x, label
    elif d_type == 'swiss_roll':
        print('* Load Swiss Roll')
        x, label = load_swiss_roll(n_samples, dev)
        return x, label
    elif d_type == 'mnist_fashion':
        print('* Load MNIST Fashion')
        x, label = load_mnist_fashion(True, n_class)
        return x, label
    elif d_type == 'mnist_digit':
        print('* Load MNIST Digit')
        x, label = load_mnist(True, n_class)
        return x, label
    elif d_type == 'mnist_digit_sk':
        print('* Load MNIST Digit SKLEARN')
        x, label = load_mnist_sk(n_class)
        return x, label
    else: 
        raise NotImplementedError
