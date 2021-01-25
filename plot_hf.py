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

    id = 14
    solver = hf.hf_solver("data/hf_{}.hdf5".format(id))
 
    fig = plt.figure()
    X = np.arange(-1.5000023, 2, 0.15)
    Y = np.arange(-1.500000054, 2, 0.15)
    Z = np.array([[solver.eval(np.array([x,y]))[0] for x in X] for y in Y])
    X, Y = np.meshgrid(X, Y)
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X,
            Y,Z)
    X = np.arange(-1.5000023, 2, 0.15)
    Y = np.arange(-1.50000054, 2, 0.15)
    Z = np.array([[solver.eval(np.array([x,y]))[1] for x in X] for y in Y])
    X, Y = np.meshgrid(X, Y)
    surf = ax.plot_surface(X,
            Y,Z)
    plt.show()
 
    
    """ PLOTTING """

    for i in range(2):
        plt.plot([tbglib.eval(k,energies,bz["k_points"])[i] for k in bz["trajectory_points"]],label = "after one iter")
    for i in range(2):
        plt.plot([tbglib.eval(k,energies_1,bz["k_points"])[i] for k in bz["trajectory_points"]],label = "after two iters")
    #for i in range(2):
    #   plt.plot([tbglib.eval(k,hf_eigenvalues,bz["k_points"])[i] for k in bz["trajectory_points"]],label = "hf_eigenvalues")
    for i in range(2):
        plt.plot([10*tbglib.eval(k,sp_energies,bz["k_points"])[i] for k in bz["trajectory_points"]],label = "10 * sp_energies")
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

