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

    id =9
    #filename = "data/coherence/hf_no_coherence.hdf5"
    filename = "data/hf_{}.hdf5".format(id)
    solver = hf.hf_solver(filename)
    print(len(solver.bz["k_points"]))
    solver.params["filling"] = -3
    solver.set_state(break_c2t=True,break_c3=False,coherent=True)
    #solver.params["epsilon"] = 12.5

    solver.valley_inv()
    print(solver.P[0])
    
    """ Solving"""

    dist =  solver.iterate_hf(True,False,False ,False)
    for m in range(500):
        dist =  solver.iterate_hf(True,False,False ,False)
        solver.valley_inv()
        print(solver.hf_energy())
    print(solver.P[0])

    filename = "data/coherence/hf_{}.hdf5".format(200+id)
    solver.save(filename)

    """ Plotting"""

    for i in range(2):
        arr = [solver.eval(k)[i] for k in solver.bz["trajectory_points"]]
        plt.plot(arr,
                label ="after" +str(m)+" hf iter"+ str(i))
    #solver.save("data/coherence/hf_{}.hdf5".format("no_coherence"))

    for i in range(2):
        plt.plot([solver.eval_sp(k)[i] for k in solver.bz["trajectory_points"]],
                label ="sp energies")
    plt.xticks(solver.bz["ticks_coords"],solver.bz["ticks_vals"])
    plt.grid()
    plt.legend()
    plt.show()
