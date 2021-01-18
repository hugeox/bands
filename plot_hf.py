import numpy as np
from math import cos,sin
import math
import time
import matplotlib.pyplot as plt
import generate_band_structure as gbs
import tbglib
import h5py
import hf
import misc

if __name__ == "__main__":
                
    """ LOADING """

    id = 3
    hf_solution = hf.read_hf_from_file("data/hf_{}.hdf5".format(id))
    for key,val in hf_solution.items():
        print("Loading key:", key)
        exec(key + '=val')
    print(model_params)
 
    #misc.check_c2t_invar(hf_eigenstates,bz,c2t_eigenvalues)
 

    
    v = hf.v_hf(bz,overlaps,model_params,P_0)
    P, energies,states = hf.iterate_hf(bz,sp_energies,overlaps, model_params,
            P_0, k_dep_filling = True)
    v_1 = v#hf.v_hf(bz,overlaps,model_params,P)
    v_final = v#hf.v_hf(bz,overlaps,model_params,P_hf)
    for k in [0]:#range(len(bz["k_points"])):
        print("V(k) c2t invar: ",np.linalg.norm(np.diag(c2t_eigenvalues[k]) @ np.conjugate(v[k])@ np.diag(np.conjugate(c2t_eigenvalues[k])) - v[k]))
        print("V(k) c2t invar after one iter:",np.linalg.norm(np.diag(c2t_eigenvalues[k]) @
                    np.conjugate(v_1[k])@
                    np.diag(np.conjugate(c2t_eigenvalues[k])) - v_1[k]))
        print("V(k) c2t invar final:",np.linalg.norm(np.diag(c2t_eigenvalues[k]) @
                    np.conjugate(v_final[k])@
                    np.diag(np.conjugate(c2t_eigenvalues[k])) - v_final[k]))
        print("2overlaps c2t invar: ",
            np.linalg.norm(np.diag(c2t_eigenvalues[k]) @np.conjugate(overlaps[4,k,k,:,:]) @ np.diag(np.conjugate(c2t_eigenvalues[k])) - overlaps[4,k,k,:,:]))
        print("2overlaps c2t invar: after one iter:",
            np.linalg.norm(np.diag(c2t_eigenvalues[k]) @np.conjugate(overlaps[4,k,k,:,:]) @ np.diag(np.conjugate(c2t_eigenvalues[k])) - overlaps[4,k,k,:,:]))

    
    for k in [bz["index_0"]]:#range(len(bz["k_points"])):
        print("HF sol",np.linalg.norm(np.diag(c2t_eigenvalues[k]) @ P_hf[k] @ np.diag(np.conjugate(c2t_eigenvalues[k])) - np.transpose(P_hf[k])))
        print("HF sol after one iter",np.linalg.norm(np.diag(c2t_eigenvalues[k]) @ P[k] @ np.diag(np.conjugate(c2t_eigenvalues[k])) - np.transpose(P[k])))
    """ PLOTTING """

    for i in range(2):
        print(hf_eigenstates[0,:,i])
        plt.plot([np.array(energies)[m,i] for m in bz["trajectory"]],'o',label = "after one iter")
    for i in range(2):
        plt.plot([np.array(hf_eigenvalues)[m,i] for m in bz["trajectory"]],label = "hf_eigenvalues")
    for i in range(2):
        plt.plot([50*np.array(sp_energies)[m,i] for m in bz["trajectory"]],
                label = "sp_energies" + str(i))
    plt.xticks(bz["ticks_coords"],bz["ticks_vals"])
    plt.legend()
    plt.show()

    for i in range(2):
        print(hf_eigenstates[0,:,i])
        plt.plot(np.array(energies)[:,i],'x',label = "after one iter")
    for i in range(2):
        plt.plot(np.array(hf_eigenvalues)[:,i] ,label = "hf_eigenvalues")
    plt.xticks(bz["ticks_coords"],bz["ticks_vals"])
    plt.legend()
    plt.show()

