import numpy as np
from math import cos,sin
import math
import time
import matplotlib.pyplot as plt
import generate_band_structure as gbs
import tbglib
import h5py


if __name__ == "__main__":
                
    id = 0
    hf_solution = hf.read_hf_from_file("data/hf_{}.hdf5".format(id))
    for key,val in hf_solution.items():
        print("Loading key:", key)
        exec(key + '=val')
    P = P_hf

    q = np.array([0,0])
    #q = bz["k_points_diff"][3]
    print("q is equal to:",q)
    k_points = bz["k_points"]
    filling = int(round(np.real(np.trace(P[0]))))
    N = len(k_points)

    H_mode=np.load("h_mode_{}.npy".format(id))
    energies, states = np.linalg.eigh(H_mode)
    print("energies:", energies[:5])
    for i in range(8):
        print("EIGSTATES", np.real(hf_eigenstates[0,:,i]))
    print("state:", states[:8-filling,0])
    dfsafdsfs
    for i in range(8):
        if np.abs(hf_eigenstates[0,7,i])**2+np.abs(hf_eigenstates[0,6,i])**2>0.5:
            print(hf_eigenstates[0,:,i])
            print(np.abs(hf_eigenstates[0,7,i])**2+np.abs(hf_eigenstates[0,6,i])**2)

    print("EIGSTATES", hf_eigenstates[0])
    V_matrix_element(1,4,5,4,7,overlaps,
            hf_eigenstates,1,0,0,1,bz)




