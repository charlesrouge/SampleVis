import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfinv

# Mann-Kendall test and associated useful functions
# In here sampled states are a 2D where each line is an ensemble member and each column is a variable

# Mann-Kendall test (precision is the number of decimals)
def mann_kendall_test(y, prec):

    n = len(y)
    x = np.int_(y*(10**prec))

    # Sign matrix and ties
    sm = np.zeros((n-1, n-1))
    for i in range(n-1):
        sm[i, i:n] = np.sign(x[i+1:n]-x[0:n-1-i])

    # Compute MK statistic
    s = np.sum(sm)

    # Count ties and their contributions to variance of the MK statistic
    [val, count] = np.unique(x, return_counts=True)
    [extent, ties] = np.unique(count, return_counts=True)
    tie_contribution = np.zeros(len(ties))
    for i in range(len(ties)):
        tie_contribution[i] = ties[i]*extent[i]*(extent[i]-1)*(2*extent[i]+5)

    # Compute the variance
    vs = (n*(n-1)*(2*n+5) - np.sum(tie_contribution))/18
    if vs < 0:
        print('WARNING: negative variance!!!')

    # Compute standard normal statistic
    z = (s-np.sign(s)) / np.sqrt(max(vs, 1))

    return z


# Same as above, but for whole sample
def test_sample(sample):

    # Local variables
    n = sample.shape[0]
    var = sample.shape[1]
    x = np.argsort(sample, axis=0)  # Ranks of the values in the ensemble, for each variable
    mk_res = np.zeros((var, var))

    # MK test results
    for i in range(var):
        reorder_lhs = np.zeros((n, var))
        for j in range(n):
            reorder_lhs[j, :] = sample[x[j, i], :]
        for v in range(var):
            if v != i:
                mk_res[i, v] = mann_kendall_test(reorder_lhs[:, v], 5)

    return mk_res


# Re-order time series:
# columns from csv file incsv
# txt inputs at directory/file "path"
# growing values of a parameter (vector x)
def re_order(path, x):

    # Parameters
    p = len(x)
    # Initialize outputs
    d = np.loadtxt(path+str(x[0])+'.txt', dtype=float, delimiter=',')
    tab = np.zeros((d.shape[0], p, d.shape[1]))

    # Fill in output matrix
    for i in range(p):
        d = np.loadtxt(path + str(x[i]) + '.txt', dtype=float, delimiter=',')
        tab[:, i, :] = d

    return tab


# MK cross-test of sample "sample", and plot
def correlation_plot(sample, var_names):

    # Local variables
    var = sample.shape[1]
    res_mat = np.zeros((var, var+1))

    # MK test results
    res_mat[:, 0:-1] = test_sample(sample)

    # Center the color scale on 0
    res_mat[0, var] = max(np.amax(res_mat), -np.amin(res_mat))
    res_mat[1, var] = -res_mat[0, var]

    # Plotting MK test results
    plt.imshow(res_mat, extent=[0, var+1, 0, var], cmap=plt.cm.bwr)

    # Plot specifications
    ax = plt.gca()
    ax.set_xlim(0, var)  # Last column only to register min and max values for colorbar
    ax.xaxis.tick_top()
    ax.set_xticks(np.linspace(0.5, var-.5, num=var))
    ax.set_xticklabels(var_names)
    ax.set_yticks(np.linspace(0.5, var-.5, num=var))
    ax.set_yticklabels(var_names[::-1])
    plt.title('Rank correlation between variables\' sampled values', size=13, y=1.07)
    plt.colorbar()
    plt.savefig('MK_cross_correlation.png')
    plt.clf()

    return res_mat[:, 0:-1]


# Discretizes value from tab depending on how they fall on a scale defined by vec
# Returns binned_tab, with the same shape as tab
# Example if vec = [0,1], binned_tab[i,j]=0 if tab[i,j]<=0, =1 if 0<tab[i,j]<=1, =2 otherwise
def binning(tab, vect):

    binned_tab = np.zeros(tab.shape)

    for i in range(len(vect)):
        binned_tab = binned_tab + 1*(tab > vect[i]*np.ones(tab.shape))

    return binned_tab


# Plot the significance of the MK-cross test of sample "sample".
def correlation_significance(sample, var_names):

    # Local variables
    var = sample.shape[1]
    res_mat = np.zeros((var, var + 1))

    # Set the thresholdsat +-95%, 99%, and 99.9% significance levels
    sig_levels = [0.9, 0.95, 0.99, 0.999]
    n_sig = len(sig_levels)
    bin_thresholds = []
    for i in range(n_sig):
        bin_thresholds.append(np.sqrt(2) * erfinv(sig_levels[i]))

    res_mat[:, 0:-1] = binning(abs(test_sample(sample)), bin_thresholds)

    # Set the color scale
    res_mat[0, var] = max(n_sig, np.amax(res_mat))

    # Common color map
    cmap = plt.cm.Greys
    cmaplist = [cmap(0)]
    for i in range(n_sig):
        cmaplist.append(cmap(int(255*(i+1)/n_sig)))
    mycmap = cmap.from_list('Custom cmap', cmaplist, n_sig+1)

    # Plot background mesh
    mesh_points = np.linspace(0.5, var - .5, num=var)
    for i in range(var):
        plt.plot(np.arange(0, var + 1), mesh_points[i]*np.ones(var+1), c='k', linewidth=0.3, linestyle=':')
        plt.plot(mesh_points[i] * np.ones(var + 1), np.arange(0, var + 1), c='k', linewidth=0.3, linestyle=':')

    # Plotting MK test results
    plt.imshow(res_mat, extent=[0, var + 1, 0, var], cmap=mycmap)



    # Plot specifications
    ax = plt.gca()
    ax.set_xlim(0, var)  # Last column only to register min and max values for colorbar
    ax.xaxis.tick_top()
    ax.set_xticks(mesh_points)
    ax.set_xticklabels(var_names)
    ax.set_yticks(mesh_points)
    ax.set_yticklabels(var_names[::-1])
    plt.title('Significance of the rank correlations', size=13, y=1.07)
    colorbar = plt.colorbar()
    colorbar.set_ticks(np.linspace(res_mat[0, var]/10, 9*res_mat[0, var]/10, num=n_sig+1))
    cb_labels = ['None']
    for i in range(n_sig):
        cb_labels.append(str(sig_levels[i]*100) + '%')
    colorbar.set_ticklabels(cb_labels)
    plt.savefig('MK_significance.png')
    plt.clf()

    return res_mat[:, 0:-1]
