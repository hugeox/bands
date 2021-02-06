import numpy as np
from math import cos,sin
import math
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import generate_band_structure as gbs
import tbglib
import h5py
import hf
import misc
from tbglib import g1,g2

if __name__ == "__main__":
                
    """ LOADING """
    id = 100
    solver = hf.hf_solver("data/hf_{}.hdf5".format(id))
    #solver.iterate_hf(True,True,False,False)
    #solver.check_v_c2t_invariance()
    centered_at = tbglib.q1-tbglib.q1 #where is the view centered, i.e. this is [0,0]
    radius = 0.3
 
    G_vals = [np.array([0,0]),g1,g2,g1+g2,-g1,-g2,-g1-g2] 
    data_new = np.concatenate(tuple([solver.hf_eigenvalues for m in range(len(G_vals))]))
    k_points_new = np.concatenate(tuple([np.array(solver.bz["k_points"])+G_vals[m]-tbglib.q1-centered_at for m in range(len(G_vals))]))
    data_new = [data_new[i] for i in range(len(k_points_new)) if np.linalg.norm(k_points_new[i])<radius]
    k_points_new = [k_points_new[i] for i in range(len(k_points_new)) if
np.linalg.norm(k_points_new[i])<radius]
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    X = [k[0] for k in k_points_new]
    Y = [k[1] for k in k_points_new]
    Z = [data_new[k][0] for k in range(len(data_new))]
    surf = ax.plot_trisurf(X,
            Y,Z)
    Z = [data_new[k][1] for k in range(len(data_new))]
    surf = ax.plot_trisurf(X,
            Y,Z)
    plt.show() 
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    X = np.arange(-1.000023, 1, 0.05)
    Y = np.arange(-1.500000054, 1.5, 0.05)
    Z = np.array([[solver.eval(np.array([x,y]))[0] for x in X] for y in Y])
    X, Y = np.meshgrid(X, Y)
    surf = ax.plot_surface(X,
            Y,Z)
    X = np.arange(-1.000023, 1, 0.05)
    Y = np.arange(-1.500000054, 1.5, 0.05)
    Z = np.array([[solver.eval(np.array([x,y]))[1] for x in X] for y in Y])
    X, Y = np.meshgrid(X, Y)
    surf = ax.plot_surface(X,
            Y,Z)
    plt.show()
 
    for i in range(2):
        plt.plot([solver.eval_sp(k)[i] for k in solver.bz["trajectory_points"]], label ="single particle")
    plt.xticks(solver.bz["ticks_coords"],solver.bz["ticks_vals"])
    plt.grid()
    plt.legend()
    plt.show()
    
    """ PLOTTING """


    for i in range(2):
        print(hf_eigenstates[0,:,i])
        plt.plot(np.array(energies)[:,i],'x',label = "after one iter")
    for i in range(2):
        plt.plot(np.array(hf_eigenvalues)[:,i] ,label = "hf_eigenvalues")
    plt.xticks(bz["ticks_coords"],bz["ticks_vals"])
    plt.legend()
    plt.show()

