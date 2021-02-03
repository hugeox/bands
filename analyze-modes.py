import numpy as np
from math import cos,sin
import math
import time
import matplotlib.pyplot as plt
import generate_band_structure as gbs
import tbglib
import h5py
import hf


if __name__ == "__main__":
                
    id = 205
    solver = hf.hf_solver("data/hf_{}.hdf5".format(id))
    q = np.array([0,0])
    filling = int(round(np.real(np.trace(solver.P[0]))))

    H_mode=np.load("data/h_mode_{}.npy".format(id))
    energies, states = np.linalg.eigh(H_mode)
    print("len:" ,len(states[:,0]))
    print("first comps" ,states[::solver.N_k,0])
    idx = np.abs(states[:,0]).argmax()
    print(states[idx,0], "index:", idx)
    k = idx%solver.N_k 
    flav = int((idx-k)/solver.N_k)
    print("eigstate",solver.hf_eigenstates[k,:,flav+1])
    print("norm eigstate",np.linalg.norm(solver.hf_eigenstates[k,:,flav+1]))
    saf
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




