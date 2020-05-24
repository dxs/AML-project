"""
Advanced test of dimensionality reduction using fashion MNIST
"""

import utils
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import manifold
import time
import argparse
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix



def create_labels():
    """
    Create label depending of the dataset used
    """
    global LABELS
    global p
    if p.data_structure == 'mnist_fashion':
        LABELS = {
            '0' : 'T-shirt',
            '1' : 'Trouser',
            '2' : 'Pullover',
            '3' : 'Dress',
            '4' : 'Coat',
            '5' : 'Sandal',
            '6' : 'Shirt',
            '7' : 'Sneaker',
            '8' : 'Bag',
            '9' : 'Ankle boot',
        }
    elif p.data_structure == 'mnist_digit':
        LABELS = {
            '0' : '0',
            '1' : '1',
            '2' : '2',
            '3' : '3',
            '4' : '4',
            '5' : '5',
            '6' : '6',
            '7' : '7',
            '8' : '8',
            '9' : '9',
        }
    else:
        raise NotImplementedError



def reduce_data(x_train, x_test):
    """
    Reduce the data and return x_train and x_test without permutations
    """

    x = np.append(x_train, x_test, axis=0)
    idx = range(x_train.shape[0])
    reverse_idx = range(x_train.shape[0], x_train.shape[0] + x_test.shape[0])

    global p
    model_m = manifold.LocallyLinearEmbedding(
        n_neighbors=p.neighbors,
        n_components=p.components,
        method='modified',
        eigen_solver=p.solver
        )
    tic = time.time()
    x_mlle = model_m.fit_transform(x)
    mlle_time = time.time() - tic
    print('** [MLLE]Done in {:.2f} seconds'.format(mlle_time))


    model_h = manifold.LocallyLinearEmbedding(
        n_neighbors=p.neighbors,
        n_components=p.components,
        method='hessian',
        eigen_solver=p.solver
        )
    tic = time.time()
    x_hlle = model_h.fit_transform(x)
    hlle_time = time.time() - tic
    print('** [HLLE]Done in {:.2f} seconds'.format(hlle_time))

    x_m_train = x_mlle[idx, :]
    x_h_train = x_hlle[idx, :]

    x_m_test = x_mlle[reverse_idx, :]
    x_h_test = x_hlle[reverse_idx, :]

    return x_m_train, x_m_test, x_h_train, x_h_test

def knn(x_train, y_train, x_test, y_test, *plot):
    """
    *plot parameter contains all the parameters to plot
    Structure of *plot : (name, )
    """
    list_n = [4, 5, 6]
    for n in list_n:
        classifier = KNeighborsClassifier(n_neighbors=n)
        classifier.fit(x_train, y_train.ravel())
        acc_knn = classifier.score(x_test, y_test.ravel())
        print('KNN {} acc : {:.02f}'.format(n, acc_knn))

def plot_reduction(x_m, x_h, y_m, y_h, *params):
    """
    plot the data after reduction of dimensionality
    """
    fig = plt.figure(figsize=(14, 7))

    fig.suptitle('MNIST_Fashion dimentionality reduction')
    ax = fig.add_subplot(121)

    #iterate for each label
    for val, label in LABELS.items():
        idx = np.argwhere(y_m == val)
        ax.scatter(x_m[idx, 0], x_m[idx, 1], label=label)
    #plt.axis('tight')
    plt.xlabel('Projection 1')
    plt.ylabel('Projection 2')
    plt.title('[MLLE]')
    plt.legend()

    ax = fig.add_subplot(122)
    #iterate for each label
    for val, label in LABELS.items():
        idx = np.argwhere(y_h == val)
        ax.scatter(x_h[idx, 0], x_h[idx, 1], label=label)

    #plt.axis('tight')
    plt.xlabel('Projection 1')
    plt.ylabel('Projection 2')
    plt.title('[HLLE]')
    plt.legend()
    plt.savefig('Results/n{}_c{}.png'.format(p.neighbors, p.n_class))
    plt.show(block=False)


def main():
    """
    Main function to handle everything
    """
    ## register params
    global p
    p = Params(
        args.data_structure,
        args.n_train,
        args.n_test,
        args.std_deviation_noise,
        args.neighbors,
        args.n_components,
        args.solver,
        args.n_class
    )
    create_labels()

    #load data
    x_raw, y_raw = utils.load_data(p.data_structure, p.n_train, p.std_dev, p.n_class)
   
    x_raw_train, x_raw_test, y_raw_train, y_raw_test = train_test_split(
        x_raw,
        y_raw,
        train_size=p.n_train,
        test_size=p.n_test
    )

    #apply dimensionality reduction
    x_m_train, x_m_test, x_h_train, x_h_test = reduce_data(x_raw_train, x_raw_test)

    y_m_train = y_raw_train
    y_m_test = y_raw_test

    y_h_train = y_raw_train
    y_h_test = y_raw_test

    # Plot reduction what is does looks like
    plot_reduction(x_m_train, x_h_train, y_m_train, y_h_train, None)


    #KNN to [MLLE]
    plot = ('MLLE')
    knn(x_m_train, y_m_train, x_m_test, y_m_test, plot)

    #KNN to [HLLE]
    plot = ('HLLE')
    knn(x_h_train, y_h_train, x_h_test, y_h_test, plot)


    


class Params():
    """
    Class to handle parameters used accross the run
    """
    def __init__(self, data_structure, n_train, n_test, std_dev, neighbors, components, solver, n_class):
        self.data_structure = data_structure
        self.n_train = n_train
        self.n_test = n_test
        self.std_dev = std_dev
        self.neighbors = neighbors
        self.components = components
        self.solver = solver
        self.n_class = n_class

if __name__ == "__main__":
    global args
    parser = argparse.ArgumentParser(description='Advanced analysis of MLLE and HLLE')

    parser.add_argument('--n_train',
                        type = int, default = 1000,
                        help = 'Number of samples used to train the structure')

    parser.add_argument('--n_test',
                        type = int, default = 50,
                        help = 'Number of samples used to test the structure')

    parser.add_argument('--neighbors',
                        type = int, default = 26,
                        help = 'Number of local neighbors')

    parser.add_argument('--n_components',
                        type = int, default = 2,
                        help = 'Number of component used to reduce')

    parser.add_argument('--std_deviation_noise',
                        type = float, default = 0,
                        help = 'Standard Deviation of noise for swiss roll generation')

    parser.add_argument('--data_structure',
                        type = str, default = 'mnist_fashion',
                        help = 'can be mnist_fashion or mnist_digit')

    parser.add_argument('--filename',
                        type = str, default='none',
                        help='filename to save under Results')

    parser.add_argument('--solver',
                        type = str, default = 'dense',
                        help = 'Solver used for the manifold reduction, it can be [auto, dense, arpack], arpack and auto can be unstable but are faster')
    parser.add_argument('--n_class',
                        type = int, default=5,
                        help='number of classes to keep (max 10)')

    args = parser.parse_args()
    main()
