import numpy as np
from math import cos,sin
import math
import time
import matplotlib.pyplot as plt
import generate_band_structure as gbs
import tbglib
import h5py
import hf

def create_state(N_k,N_f,filling,break_c2t=False,break_c3 = False, coherence=False):
    P_1=[]
    P = np.zeros((N_f,N_f),dtype = complex)
    if  coherence == True:
        if filling+4%2!=0:
            print("Warning, need even filling for coherence!")
        for i in range(int((filling+4)/2)):
            P[4*i:4*i+4,4*i:4*i+4] = np.kron(0.5*1/math.sqrt(2)*(tbglib.sx+tbglib.sy), \
                tbglib.sy)+ 1/2*np.kron(tbglib.s0,tbglib.s0 )
        P1 = P
        P2 = P
        print(P)
    else:
        m = np.zeros((int(N_f/2)),dtype = complex)
        m[:filling+4]=1
        if break_c2t:
            P1 =  np.kron(np.diag(m),1/2*(tbglib.s0 + tbglib.sy))
        else:
            P1 =  np.kron(np.diag(m),1/2*(tbglib.s0 + tbglib.sz))
        P2 =  np.kron(np.diag(m),1/2*(tbglib.s0 - tbglib.sz))
    for k in range(N_k):
        if k%7==0 and break_c3:
            P_1.append(P2.copy())
        else:
            P_1.append(P1.copy())     
    return P_1
if __name__ == "__main__":
                
    """ LOADING """

    id = 5
    solver = hf.hf_solver("data/hf_{}.hdf5".format(id))
    solver.params["description"] = "HF,  c2t_breaking "
    solver.params["filling"] = 0
    P = create_state(solver.N_k,solver.params["N_f"],solver.params["filling"],break_c2t= True,
                        break_c3 = False, coherence = False)
    solver.params["epsilon"] = 0.5*1/0.06  
    solver.reset_P(P)

    for m in range(240):
        dist =  solver.iterate_hf(True,True,False ,False)
        print(solver.hf_energy())
    for i in range(2):
        arr = [solver.eval(k)[i] for k in solver.bz["trajectory_points"]]
        plt.plot(arr,
                label ="after" +str(m)+" hf iter"+ str(i))
    #solver.save("data/hf_{}.hdf5".format(100+id))
    solver.save("data/coherence/hf_{}.hdf5".format("no_coherence_2"))

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
