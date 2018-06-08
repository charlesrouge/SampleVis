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
        y[i,:] = 1*(np.sum(1*(binned_sample == i+1), axis=0) > 0)

    proj = np.sum(y, axis=0)/ n

    plt.bar(np.arange(dim), proj)
    ax = plt.gca()
    plt.ylim(0, max(1, 1.01*np.amax(proj)))
    plt.xticks(np.arange(dim), var_names)
    plt.ylabel('Coverage of axis')
    plt.savefig('1D_coverage_index.png')
    plt.clf()

    return proj


def projection_2D(sample, var_names):

    [n, dim] = sample.shape

    for i in range(dim):
        for j in range(dim):
            plt.subplot(dim, dim, i*dim + j + 1)
            plt.scatter(sample[:,i], sample[:,j], .1)
            plt.xticks()
            plt.yticks()
    plt.savefig('draft.png')
    plt.clf()

    return None
