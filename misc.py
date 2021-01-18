import numpy as np


def check_c2t_invar(hf_eigenstates,bz,c2t_eigenvalues):
    for k in range(len(bz["k_points"])):
        for i in range(8):
            print("k, i are:", k, i)
            for j in range(8):
                if abs(hf_eigenstates[k][j,i])>1e-18:
                    print((np.diag(c2t_eigenvalues[k])@np.conjugate(hf_eigenstates[k][:,i]))[j]/hf_eigenstates[k][j,i])
