import pynolh
from SALib.sample import saltelli
from SALib.sample import sobol_sequence
import mann_kendall
import pearson
from scipy.special import erf
import numpy as np

"""
dim = 6
m, q, r = pynolh.params(dim)
print(r)
conf = range(q)
print(conf)
remove = range(dim - r, dim)
# nolh = pynolh.nolh(conf, remove)
"""

n = 7
names = []
for i in range(n):
    names.append('x' + str(i+1))

problem = {
    'num_vars': n,
    'names': names,
    'bounds': [[-1, 1]]*n
}

param_values = sobol_sequence.sample(128, problem['num_vars'])
# param_values = saltelli.sample(problem, 8)
# z = mann_kendall.mann_kendall_test(param_values[:,6],5)

print(np.matlib.eye(3))
#x = mann_kendall.correlation_plot(param_values, problem['names'])
# y = mann_kendall.correlation_significance(param_values, problem['names'])


# x = pearson.correlation_plots(param_values, problem['names'])