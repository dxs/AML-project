import utils
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import manifold
import time
import argparse

parser = argparse.ArgumentParser(description='Advanced analysis of MLLE and HLLE')

parser.add_argument('--n_samples',
                    type = int, default = 5000,
                    help = 'Number of samples used to train')

parser.add_argument('--neighbors',
                    type = int, default = 12,
                    help = 'Number of local neighbors')

parser.add_argument('--n_components',
                    type = int, default = 2,
                    help = 'Number of component used to reduce')

parser.add_argument('--data_structure',
                    type = str, default = 'mnist-fashion',
                    help = 'can be swiss_roll, s_curve, mnist')

parser.add_argument('--filename',
                    type = str, default='none',
                    help='filename to save under Results')

parser.add_argument('--solver',
                    type = str, default = 'auto',
                    help = 'Solver used for the manifold reduction, it can be [auto, dense, arpack], arpack and auto can be unstable but are faster')

args = parser.parse_args()

#load data
x, color = utils.load_data(args.data_structure, args.n_samples, args.std_deviation_noise)


print('** Compute MLLE')
tic = time.time()
x_mlle, err_mlle = manifold.locally_linear_embedding(x, n_neighbors=args.neighbors, n_components=args.n_components, method='modified', eigen_solver=args.solver)
mlle_time = time.time() - tic
print('** Done with reconstruction error : {:.2f} in {:.2f} seconds'.format(err_mlle, mlle_time))


print('** Compute HLLE')
tic = time.time()
x_hlle, err_hlle = manifold.locally_linear_embedding(x, n_neighbors=args.neighbors, n_components=args.n_components, method='hessian', eigen_solver=args.solver)
hlle_time = time.time() - tic
print('** Done with reconstruction error : {:.2f} in {:.2f} seconds'.format(err_hlle, hlle_time))

# Plot results
fig = plt.figure(figsize=(15, 5))
fig.suptitle('{} - {} neighbors - std {:.02f}'.format(args.data_structure, args.neighbors, args.std_deviation_noise))
ax = fig.add_subplot(131, projection='3d')
ax.view_init(15, -80)
ax.scatter(x[:, 0], x[:, 1], x[:, 2], c=color, cmap=plt.cm.Spectral)

ax.set_title("Original data")

ax = fig.add_subplot(132)
ax.scatter(x_mlle[:, 0], x_mlle[:, 1], c=color, cmap=plt.cm.Spectral)
plt.axis('tight')
plt.xlabel('Projection 1')
plt.ylabel('Projection 2')
plt.title('Projected data [MLLE] [{:.2f}s, err {:.2f}]'.format(mlle_time, err_mlle))

ax = fig.add_subplot(133)
ax.scatter(x_hlle[:, 0], x_hlle[:, 1], c=color, cmap=plt.cm.Spectral)
plt.axis('tight')
plt.xlabel('Projection 1')
plt.title('Projected data [HLLE] [{:.2f}s, err {:.2f}])'.format(hlle_time, err_hlle))
plt.savefig('Results/{}.png'.format(args.filename))

plt.show(block=True)
