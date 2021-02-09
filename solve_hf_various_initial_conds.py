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

    id = 5
    #filename = "data/coherence/hf_no_coherence.hdf5"
    filename = "data/hf_5.hdf5"
    solver = hf.hf_solver(filename)
    solver.params["filling"] = 0
    solver.set_state(break_c2t=True,break_c3=False,coherent=False)
    solver.reset_P(solver.P)
    solver.params["epsilon"] = 1/0.06 *5

    dist =  solver.iterate_hf(True,True,False ,False)
    for m in range(120):
        dist =  solver.iterate_hf(True,True,False ,False)
        tbglib.valley_inv(solver.P)
        #print(solver.P[2])
        print(solver.hf_energy())
    for i in range(2):
        arr = [solver.eval(k)[i] for k in solver.bz["trajectory_points"]]
        plt.plot(arr,
                label ="after" +str(m)+" hf iter"+ str(i))
    #solver.save("data/hf_{}.hdf5".format(id))
    solver.save("data/coherence/hf_{}.hdf5".format("no_coherence"))

    for i in range(2):
        plt.plot([solver.eval_sp(k)[i] for k in solver.bz["trajectory_points"]],
                label ="sp energies")
    plt.xticks(solver.bz["ticks_coords"],solver.bz["ticks_vals"])
    plt.grid()
    plt.legend()
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
