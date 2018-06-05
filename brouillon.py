from SALib.sample import saltelli
import mann_kendall

problem = {
    'num_vars': 8,
    'names': ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8'],
    'bounds': [[-1, 1]]*8
}

param_values = saltelli.sample(problem, 200)

# x = mann_kendall.correlation_plot(param_values, problem['names'])
x = mann_kendall.correlation_significance(param_values, problem['names'])
