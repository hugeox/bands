import numpy as np
from math import cos,sin
import math
import time
import matplotlib.pyplot as plt
import generate_band_structure as gbs
import tbglib
import h5py
import hf

def create_state(N_f):
    P = np.zeros((N_f,N_f),dtype = complex)
    states = 1/2*(tbglib.s0 + tbglib.sy)
    P[:2,:2]=states
    return P
if __name__ == "__main__":
                
    """ LOADING """

    id = 5
    solver = hf.hf_solver("data/hf_{}.hdf5".format(id))
    solver.params["epsilon"] = 1/0.06 
    solver.params["description"] = "HF,  break c2t "
    P_1=[]
    

    for k in range(solver.N_k):
        P_k = create_state(solver.params["N_f"])
        if k>solver.N_k/2 and False:
            P_k[:2,:2] = 1/2*(tbglib.s0 - tbglib.sz)
        #P_k=np.diag(a)
        P_1.append(P_k.copy())
    solver.reset_P(P_1)
    print(P_1[-1])

    solver.iterate_hf(True,True,False ,True)
    for m in range(80):
        dist =  solver.iterate_hf(True,True,False ,False)
        if m%20==0:
            for i in range(2):
                arr = [solver.eval(k)[i] for k in solver.bz["trajectory_points"]]
                plt.plot(arr,
                        label ="after" +str(m)+" hf iter"+ str(i))

    id = 200+id  
    solver.save("data/hf_{}.hdf5".format(id))
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
0+id+idd
