import numpy as np
import matplotlib.pyplot as plt
import util


# Assumes bounds of sample are [0,1]**n
def projection_1D(sample, var_names):

    [n, dim] = sample.shape
    # bounds = [[0, 1]]*n
    y = np.zeros(sample.shape)

    z_int = np.linspace(0, 1, num=n + 1)
    binned_sample = util.binning(sample, z_int)

    # for d in range(dim):

    for i in range(n):
        y[i, :] = 1*(np.sum(1*(binned_sample == i+1), axis=0) > 0)

    proj = np.sum(y, axis=0) / n

    plt.bar(np.arange(dim), proj)
    plt.ylim(0, max(1, 1.01*np.amax(proj)))
    plt.xticks(np.arange(dim), var_names)
    plt.ylabel('Coverage of axis')
    plt.savefig('1D_coverage_index.png')
    plt.clf()

    return proj


# Plots the sample projected on each 2D plane
def projection_2D(sample, var_names):

    dim = sample.shape[1]

    for i in range(dim):
        for j in range(dim):
            plt.subplot(dim, dim, i*dim + j + 1)
            plt.scatter(sample[:, j], sample[:, i], .1)
            if j == 0:
                plt.ylabel(var_names[i])
            if i == dim-1:
                plt.xlabel(var_names[j])

            plt.xticks([])
            plt.yticks([])
    plt.savefig('2D-projections.png')
    plt.clf()

    return None
