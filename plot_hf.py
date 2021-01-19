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

    id = 8
    hf_solution = hf.read_hf_from_file("data/hf_{}.hdf5".format(id))
    for key,val in hf_solution.items():
        print("Loading key:", key)
        exec(key + '=val')
    print(model_params)
 
    print("energies at gamma", tbglib.eval(np.array([-2,-2]),hf_eigenvalues,bz["k_points"]))
    #misc.check_c2t_invar(hf_eigenstates,bz,c2t_eigenvalues)
    for i in range(2):
        plt.plot([tbglib.eval(k,hf_eigenvalues,bz["k_points"])[i] for k in bz["trajectory_points"]],label = "hf_eigenvalues")
    plt.xticks(bz["ticks_coords"],bz["ticks_vals"])
    plt.legend()
    plt.show()
    fig = plt.figure()

    P, energies,states = hf.iterate_hf(bz,sp_energies,overlaps, model_params,
            P_0, k_dep_filling = False)
    X = np.arange(-2.0000023, 2, 0.15)
    Y = np.arange(-2.00000054, 2, 0.15)
    Z = np.array([[tbglib.eval(np.array([x,y]),energies,bz["k_points"])[0] for x in X] for y in Y])
    X, Y = np.meshgrid(X, Y)
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X,
            Y,Z)
    X = np.arange(-2.0000023, 2, 0.15)
    Y = np.arange(-2.00000054, 2, 0.15)
    Z = np.array([[tbglib.eval(np.array([x,y]),energies,bz["k_points"])[1] for x in X] for y in Y])
    X, Y = np.meshgrid(X, Y)
    surf = ax.plot_surface(X,
            Y,Z)
    plt.show()
 

    
    v = hf.v_hf(bz,overlaps,model_params,P_0)
    v_1 = hf.v_hf(bz,overlaps,model_params,P)
    v_final = hf.v_hf(bz,overlaps,model_params,P_hf)
    P, energies_1,states = hf.iterate_hf(bz,sp_energies,overlaps, model_params,
            P, k_dep_filling = False)
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
        plt.plot([tbglib.eval(k,energies,bz["k_points"])[i] for k in bz["trajectory_points"]],label = "after one iter")
    for i in range(2):
        plt.plot([tbglib.eval(k,energies_1,bz["k_points"])[i] for k in bz["trajectory_points"]],label = "after two iters")
    for i in range(2):
        plt.plot([tbglib.eval(k,hf_eigenvalues,bz["k_points"])[i] for k in bz["trajectory_points"]],label = "hf_eigenvalues")
    for i in range(2):
        plt.plot([50*tbglib.eval(k,sp_energies,bz["k_points"])[i] for k in bz["trajectory_points"]],label = "50 * sp_energies")
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

