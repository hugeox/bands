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
                
    """ LOADING """

    id = 0
    hf_solution = hf.read_hf_from_file("data/hf_{}.hdf5".format(id))
    for key,val in hf_solution.items():
        print("Loading key:", key)
        exec(key + '=val')
    print(model_params)
    print(P_hf[0])
 
 

    
    v = hf.v_hf(bz,overlaps,model_params,P_0)
    for k in range(len(bz["k_points"])):
        print("V(k) c2t invar: ",np.linalg.norm(np.diag(c2t_eigenvalues[k]) @ np.conjugate(v[k])@ np.diag(np.conjugate(c2t_eigenvalues[k])) - v[k]))
        print("2overlaps c2t invar: ",
            np.linalg.norm(np.diag(c2t_eigenvalues[k]) @np.conjugate(overlaps[4,k,k,:,:]) @ np.diag(np.conjugate(c2t_eigenvalues[k])) - overlaps[4,k,k,:,:]))

    
    P, energies,states = hf.iterate_hf(bz,sp_energies,overlaps, model_params, P_0)
    for k in range(len(bz["k_points"])):
        print("HF sol",np.linalg.norm(np.diag(c2t_eigenvalues[k]) @ np.transpose(P_hf[k])@ np.diag(np.conjugate(c2t_eigenvalues[k])) - P_hf[k]))
        print("HF sol after one iter",np.linalg.norm(np.diag(c2t_eigenvalues[k]) @ np.transpose(P[k])@ np.diag(np.conjugate(c2t_eigenvalues[k])) - P[k]))
    """ PLOTTING """

    for i in range(2):
        print(hf_eigenstates[0,:,i])
        plt.plot([np.array(energies)[m,i] for m in bz["trajectory"]])
    for i in range(2):
        plt.plot([np.array(hf_eigenvalues)[m,i] for m in bz["trajectory"]])
    for i in range(2):
        plt.plot([50*np.array(sp_energies)[m,i] for m in bz["trajectory"]])
    plt.xticks(bz["ticks_coords"],bz["ticks_vals"])
    plt.legend()
    plt.show()

