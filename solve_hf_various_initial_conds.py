import numpy as np
from math import cos,sin
import math
import time
import matplotlib.pyplot as plt
import generate_band_structure as gbs
import tbglib
import h5py
import hf

def create_state():
    states = 1/2*(tbglib.s0 + tbglib.sz)
    m = np.zeros(2,dtype = complex)
    m[:1]=1
    P_new = np.conjugate(states) @\
                 np.diag(m) @  np.transpose(states)
    return states
if __name__ == "__main__":
                
    """ LOADING """

    id = 15
    solver = hf.hf_solver("data/hf_{}.hdf5".format(id))
    solver.params["epsilon"] = 1/0.06 *100
    P_1=[]
    a = np.array([1+0j,0,])
    print(solver.eval_sp(tbglib.q1))

    for k in range(solver.N_k):
        P_k = create_state()
        #P_k=np.diag(a)
        P_1.append(P_k.copy())
    solver.reset_P(P_1)

    for m in range(50):
        solver.iterate_hf(True,True, True)
        if m%10==0:
            for i in range(2):
                plt.plot([solver.eval(k)[i] for k in solver.bz["trajectory_points"]],
                        label ="after" +str(m)+" hf iter"+ str(i))

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

