import numpy as np


# Discretizes value from tab depending on how they fall on a scale defined by vec
# Returns binned_tab, with the same shape as tab
# Example if vec = [0,1], binned_tab[i,j]=0 if tab[i,j]<=0, =1 if 0<tab[i,j]<=1, =2 otherwise
def binning(tab, vect):

    binned_tab = np.zeros(tab.shape)

    for i in range(len(vect)):
        binned_tab = binned_tab + 1*(tab > vect[i]*np.ones(tab.shape))

    return binned_tab
