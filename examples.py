import utils
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import manifold
import time
import argparse

parser = argparse.ArgumentParser(description='Swiss Roll example')

parser.add_argument('--n_samples',
                    type = int, default = 2000,
                    help = 'Number of samples used to create the structure')

parser.add_argument('--n_neighbors',
                    type = int, default = 12,
                    help = 'Number of local neighbors')

parser.add_argument('--n_components',
                    type = int, default = 2,
                    help = 'Number of component used to reduce')

parser.add_argument('--std_deviation_noise',
                    type = float, default = 0.1,
                    help = 'Standard Deviation of noise for swiss roll generation')

parser.add_argument('--swiss_roll',
                    action = 'store_true', default = True,
                    help = 'Use swiss_roll, used by default')

parser.add_argument('--s_curve',
                    action = 'store_true', default = False,
                    help = 'Use s-curve')

parser.add_argument('--solver',
                    type = str, default = 'auto',
                    help = 'Solver used for the manifold reduction, it can be [auto, dense, arpack], arpack and auto can be unstable but are faster')

args = parser.parse_args()


# Load data
if args.s_curve is True:
    print('* Load S Curve')
    x, color = utils.load_s_curve(args.n_samples, args.std_deviation_noise)
else:
    print('* Load Swiss Roll')
    x, color = utils.load_swiss_roll(args.n_samples, args.std_deviation_noise)

print('** Compute LLE')
tic = time.time()
x_lle, err_lle = manifold.locally_linear_embedding(x, n_neighbors=args.n_neighbors, n_components=args.n_components, eigen_solver=args.solver)
toc = time.time()
print('** Done with reconstruction error : {:.2f} in {:.2f} seconds'.format(err_lle, toc-tic))

print('** Compute MLLE')
tic = time.time()
x_mlle, err_mlle = manifold.locally_linear_embedding(x, n_neighbors=args.n_neighbors, n_components=args.n_components, method='modified', eigen_solver=args.solver)
toc = time.time()
print('** Done with reconstruction error : {:.2f} in {:.2f} seconds'.format(err_mlle, toc-tic))

#check if Hessian is possible:
if args.n_neighbors <= (args.n_components *(1 + (args.n_components + 1) / 2) ) : 
    print('** WARNING n_neighbors and n_components are not compatible with HLLE')
else:
    print('** Compute HLLE')
    tic = time.time()
    x_hlle, err_hlle = manifold.locally_linear_embedding(x, n_neighbors=args.n_neighbors, n_components=args.n_components, method='hessian', eigen_solver=args.solver)
    toc = time.time()
    print('** Done with reconstruction error : {:.2f} in {:.2f} seconds'.format(err_hlle, toc-tic))

# Plot results
fig = plt.figure()

ax = fig.add_subplot(221, projection='3d')
ax.scatter(x[:, 0], x[:, 1], x[:, 2], c=color, cmap=plt.cm.Spectral)

ax.set_title("Original data")

ax = fig.add_subplot(222)
ax.scatter(x_lle[:, 0], x_lle[:, 1], c=color, cmap=plt.cm.Spectral)
plt.axis('tight')
plt.xticks([]), plt.yticks([])
plt.title('Projected data [LLE]')

ax = fig.add_subplot(223)
ax.scatter(x_mlle[:, 0], x_mlle[:, 1], c=color, cmap=plt.cm.Spectral)
plt.axis('tight')
plt.xticks([]), plt.yticks([])
plt.title('Projected data [MLLE]')

ax = fig.add_subplot(224)
ax.scatter(x_hlle[:, 0], x_hlle[:, 1], c=color, cmap=plt.cm.Spectral)
plt.axis('tight')
plt.xticks([]), plt.yticks([])
plt.title('Projected data [HLLE]')
plt.show()
