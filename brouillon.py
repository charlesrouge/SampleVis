from SALib.sample import sobol_sequence
from SALib.sample import latin
from SALib.sample import saltelli
from SALib.sample.morris import local
import mann_kendall
import pearson
from scipy.special import erf
import numpy as np
import util
import projection_properties

"""
dim = 6
m, q, r = pynolh.params(dim)
print(r)
conf = range(q)
print(conf)
remove = range(dim - r, dim)
# nolh = pynolh.nolh(conf, remove)
"""

n = 100
dim = 8
names = []
for i in range(dim):
    names.append('x' + str(i+1))

problem = {
    'num_vars': dim,
    'names': names,
    'bounds': [[0, 1]]*dim
}

# param_values = sobol_sequence.sample(n, problem['num_vars'])
param_values = latin.sample(problem, n)
# param_values = saltelli.sample(problem, 100)

print(param_values.shape)

x = projection_properties.projection_1D(param_values, problem['names'])
projection_properties.projection_2D(param_values, problem['names'])


[z_mk, pval_mk] = mann_kendall.test_sample(param_values)
[rho, pval_pr] = pearson.test_sample(param_values)

util.correlation_plots(z_mk, pval_mk, 'Mann-Kendall', problem['names'])
util.correlation_plots(rho, pval_pr, 'Pearson', problem['names'])

# x = pearson.correlation_plots(param_values, problem['names'])
