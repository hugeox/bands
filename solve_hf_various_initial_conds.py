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

    id = 8
    hf_solution = hf.read_hf_from_file("data/hf_{}.hdf5".format(id))
    for key,val in hf_solution.items():
        print("Loading key:", key)
        exec(key + '=val')
    print(model_params)
 
    P_1=[]
    for k in range(N):
        a = np.array([1+0j,0,])
        states = 1/math.sqrt(2)*np.transpose([np.conjugate(c2t_eigenvalues[k][:2]),[0,0]])
        #states = 1/math.sqrt(2)*np.transpose([np.sqrt(c2t_eigenvalues[k][:2]),[0,0]])
        P_k=np.diag(a)
        #P_k[:2,:2]= np.conjugate(states) @\
        #        np.diag(a)[:2,:2] @  np.transpose(states)
        P_1.append(P_k)

 
    zero = bz["index_0"]
    
    k = 0
    print("P c2t invariance:",np.linalg.norm(np.diag(c2t_eigenvalues[k]) @ P_1[k] @ np.diag(np.conjugate(c2t_eigenvalues[k])) - np.transpose(P_1[k])))
    P, energies,states = hf.iterate_hf(bz,sp_energies,overlaps, model_params,
            P_1, k_dep_filling = False)
    print("P c2t invariance:",np.linalg.norm(np.diag(c2t_eigenvalues[k]) @ P[k] @ np.diag(np.conjugate(c2t_eigenvalues[k])) - np.transpose(P[k])))
    for m in range(50):
        P_old = P.copy()
        P, hf_eig,hf_states = hf.iterate_hf(bz,sp_energies,overlaps,
                model_params, P_old,False)
        print(m,"P c2t invariance:",np.linalg.norm(np.diag(c2t_eigenvalues[k]) @ P[k] @ np.diag(np.conjugate(c2t_eigenvalues[k])) - np.transpose(P[k])))
        print(np.linalg.norm(np.array(P).ravel()-np.array(P_old).ravel()))

    for k in range(len(bz["k_points"])):
        for i in range(2):
            if np.linalg.norm(states[k][2:,i])>0.01:
                print(states[k][:,i])

        print("V(k) c2t invar with P_0: ",np.linalg.norm(np.diag(c2t_eigenvalues[k]) @ np.conjugate(v[k])@ np.diag(np.conjugate(c2t_eigenvalues[k])) - v[k]))
        print("P_0 c2t invar",np.linalg.norm(np.diag(c2t_eigenvalues[k]) @ P_1[k] @\
            np.diag(np.conjugate(c2t_eigenvalues[k])) - np.transpose(P_1[k])))
        print("HF sol after many iter",np.linalg.norm(np.diag(c2t_eigenvalues[k]) @ P[k] @ np.diag(np.conjugate(c2t_eigenvalues[k])) - np.transpose(P[k])))


    for i in range(2):
        plt.plot(np.array(energies)[:,i] ,label =
        "after one hf iter"+ str(i))
    plt.show()

    """ PLOTTING """
    for i in range(2):
        print(states[0][:,i])
        plt.plot([np.array(energies)[m,i] for m in bz["trajectory"]],label =
        "after one hf iter"+ str(i))
    for i in range(2):
        plt.plot([np.array(hf_eig)[m,i] for m in bz["trajectory"]],
                label = "final" + str(i))
    for i in range(2):
        plt.plot([50*np.array(sp_energies)[m,i] for m in bz["trajectory"]],
                label = "50* sp_energies" + str(i))
    plt.xticks(bz["ticks_coords"],bz["ticks_vals"])
    plt.legend()
    plt.show()

