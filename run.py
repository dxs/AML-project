import os
from subprocess import Popen

def execute(n_samples, neighbors, dev, structure, filename, solver, components=2, MNIST = False):
    if MNIST is False:
        Popen('python basics.py --n_samples {:d} --neighbors {:d} --std_deviation_noise {} --data_structure {} --solver {} --filename {} --n_components {}'.format(n_samples, neighbors, dev, structure, solver, filename, components))
    elif MNIST is True:
        pass

#swiss rolls
execute(2000, 20, 0.0, 'swiss_roll', 'swiss_roll_STD0_N20', 'dense')
execute(2000, 20, 0.4, 'swiss_roll', 'swiss_roll_STD0.4_N20', 'dense')
execute(2000, 35, 0.0, 'swiss_roll', 'swiss_roll_STD0_N35', 'dense')

# s curve
execute(2000, 20, 0.0, 's_curve', 's_curve_STD0_N20', 'dense')
execute(2000, 20, 0.2, 's_curve', 's_curve_STD0.2_N20', 'dense')
execute(2000, 35, 0.0, 's_curve', 's_curve_STD0_N35', 'dense')


