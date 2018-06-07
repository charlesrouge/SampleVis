import numpy as np
import *

# Discretizes value from tab depending on how they fall on a scale defined by vec
# Returns binned_tab, with the same shape as tab
# Example if vec = [0,1], binned_tab[i,j]=0 if tab[i,j]<=0, =1 if 0<tab[i,j]<=1, =2 otherwise
def binning(tab, vect):

    binned_tab = np.zeros(tab.shape)

    for i in range(len(vect)):
        binned_tab = binned_tab + 1*(tab > vect[i]*np.ones(tab.shape))

    return binned_tab


# Get and plot results for within-sample correlation test
# 1) test results
# 2) statistical significance
def correlation_plots(sample, var_names):

    # Local variables
    nvar = len(var_names)


    # Pearson test results
    [rho, pval] = test_sample(sample)
    pval = 1 - (pval + np.matlib.eye(nvar))  # Transformation convenient for plotting below

    ###################################################
    # Figure 1: correlations

    # Matrix to plot
    res_mat = np.zeros((nvar, nvar + 1))
    res_mat[:, 0:-1] = rho

    # Center the color scale on 0
    res_mat[0, nvar] = max(np.amax(rho), -np.amin(rho))
    res_mat[1, nvar] = -res_mat[0, nvar]

    # Plotting Pearson test results
    plt.imshow(res_mat, extent=[0, nvar+1, 0, nvar], cmap=plt.cm.bwr)

    # Plot specifications
    ax = plt.gca()
    ax.set_xlim(0, nvar)  # Last column only to register min and max values for colorbar
    ax.xaxis.tick_top()
    ax.set_xticks(np.linspace(0.5, nvar - .5, num=nvar))
    ax.set_xticklabels(var_names)
    ax.set_yticks(np.linspace(0.5, nvar - .5, num=nvar))
    ax.set_yticklabels(var_names[::-1])
    plt.title('Rank correlation between variables\' sampled values', size=13, y=1.07)
    plt.colorbar()
    plt.savefig('Pearson_cross_correlation.png')
    plt.clf()

    ###################################################
    # Figure 2: correlations

    # Matrix to plot
    res_mat = np.zeros((nvar, nvar + 1))

    # Set the thresholds at +-95%, 99%, and 99.9% significance levels
    bin_thresholds = [0.9, 0.95, 0.99, 0.999]
    n_sig = len(bin_thresholds)
    res_mat[:, 0:-1] = util.binning(pval, bin_thresholds)

    # Set the color scale
    res_mat[0, nvar] = n_sig

    # Common color map
    cmap = plt.cm.Greys
    cmaplist = [cmap(0)]
    for i in range(n_sig):
        cmaplist.append(cmap(int(255 * (i + 1) / n_sig)))
    mycmap = cmap.from_list('Custom cmap', cmaplist, n_sig + 1)

    # Plot background mesh
    mesh_points = np.linspace(0.5, nvar - .5, num=nvar)
    for i in range(nvar):
        plt.plot(np.arange(0, nvar + 1), mesh_points[i] * np.ones(nvar + 1), c='k', linewidth=0.3, linestyle=':')
        plt.plot(mesh_points[i] * np.ones(nvar + 1), np.arange(0, nvar + 1), c='k', linewidth=0.3, linestyle=':')

    # Plotting MK test results
    plt.imshow(res_mat, extent=[0, nvar + 1, 0, nvar], cmap=mycmap)

    # Plot specifications
    ax = plt.gca()
    ax.set_xlim(0, nvar)  # Last column only to register min and max values for colorbar
    ax.xaxis.tick_top()
    ax.set_xticks(mesh_points)
    ax.set_xticklabels(var_names)
    ax.set_yticks(mesh_points)
    ax.set_yticklabels(var_names[::-1])
    plt.title('Significance of the rank correlations', size=13, y=1.07)
    colorbar = plt.colorbar()
    colorbar.set_ticks(np.linspace(res_mat[0, nvar] / 10, 9 * res_mat[0, nvar] / 10, num=n_sig + 1))
    cb_labels = ['None']
    for i in range(n_sig):
        cb_labels.append(str(bin_thresholds[i] * 100) + '%')
    colorbar.set_ticklabels(cb_labels)
    plt.savefig('Pearson_significance.png')
    plt.clf()

    return [rho, pval]