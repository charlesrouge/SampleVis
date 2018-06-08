import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfinv
from scipy.stats import pearsonr
import util

# Correlation Pearson test for whole sample. Outputs are:
# the Pearson statistic rho
# the p-value pval
def test_sample(sample):

    # Local variables
    var = sample.shape[1]
    rho = np.zeros((var, var))
    pval = np.zeros((var, var))

    # MK test results
    for i in range(var):
        for v in np.arange(i+1, var):
            [rho[i, v], pval[i, v]] = pearsonr(sample[:, i], sample[:, v])
            [rho[v, i], pval[v, i]] = [rho[i, v], pval[i, v]]

    return [rho, pval]
