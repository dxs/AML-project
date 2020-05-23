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






def reduce_data(input):
    """
    Reduce the data and return x_train and x_test without permutations
    """
    global p
    tic = time.time()
    x_mlle, err_mlle = manifold.locally_linear_embedding(input, n_neighbors=p.neighbors, n_components=p.components, method='modified', eigen_solver=p.solver)
    mlle_time = time.time() - tic


    tic = time.time()
    x_hlle, err_hlle = manifold.locally_linear_embedding(input, n_neighbors=p.neighbors, n_components=p.components, method='hessian', eigen_solver=p.solver)
    hlle_time = time.time() - tic
    print('** [MLLE]Done with reconstruction error : {:.2f} in {:.2f} seconds'.format(err_mlle, mlle_time))
    print('** [HLLE]Done with reconstruction error : {:.2f} in {:.2f} seconds'.format(err_hlle, hlle_time))

    return x_mlle, x_hlle



def main():
    ## register params
    global p 
    p = Params(args.data_structure, args.n_samples, args.std_deviation_noise, args.neighbors, args.n_components, args.solver)

    #load data
    x_raw, y = utils.load_data(p.data_structure, p.n_samples, p.std_dev)
    
    x_raw_train, x_raw_test, y_raw_train, y_raw_test = train_test_split(x_raw, y, train_size=p.n_samples, test_size=1000)
    x_used = np.append(x_raw_train, x_raw_test, axis=0)

    #apply dimensionality reduction 
    x_m, x_h = reduce_data(x_used)

    

    


class Params():
    def __init__(self, data_structure, n_samples, std_dev, neighbors, components, solver):
        self.data_structure = data_structure
        self.n_samples = n_samples
        self.std_dev = std_dev
        self.neighbors = neighbors
        self.components = components
        self.solver = solver

if __name__ == "__main__":

    global args
    parser = argparse.ArgumentParser(description='Advanced analysis of MLLE and HLLE')

    parser.add_argument('--n_samples',
                        type = int, default = 2000,
                        help = 'Number of samples used to create the structure')

    parser.add_argument('--neighbors',
                        type = int, default = 12,
                        help = 'Number of local neighbors')

    parser.add_argument('--n_components',
                        type = int, default = 2,
                        help = 'Number of component used to reduce')

    parser.add_argument('--std_deviation_noise',
                        type = float, default = 0.1,
                        help = 'Standard Deviation of noise for swiss roll generation')

    parser.add_argument('--data_structure',
                        type = str, default = 'mnist_fashion',
                        help = 'can be mnist_fashion')

    parser.add_argument('--filename',
                        type = str, default='none',
                        help='filename to save under Results')

    parser.add_argument('--solver',
                        type = str, default = 'auto',
                        help = 'Solver used for the manifold reduction, it can be [auto, dense, arpack], arpack and auto can be unstable but are faster')

    args = parser.parse_args()
    main()