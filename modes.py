import numpy as np
from math import cos,sin
import math
import time
import matplotlib.pyplot as plt
import generate_band_structure as gbs
import tbglib
import h5py
import hf


def V_matrix_element(g,k1,k2,k3,k4,i,j,k,l,overlaps,
        hf_eigenstates,bz):
    U = hf_eigenstates
    element = (tbglib.dagger(U[k1]) @ overlaps[g,k1,k2,:,:] @ U[k2])[i,j] *\
                (tbglib.dagger(U[k3]) @ overlaps[bz["G_neg_indices"][g],k3,k4,:,:] @ U[k4])[k,l] *\
                hf.V_coulomb(bz["k_points"][k2]-bz["k_points"][k1]+bz["G_values"][g])
    return element

def index_kplusq(bz,index_k,q):
    kplusq = bz["k_points"][index_k]+q
    k_inbz, G = tbglib.decompose(kplusq)

    idx = (np.linalg.norm(np.array(bz["k_points"]) - k_inbz ,axis=1)).argmin()
    if idx!=index_k:
        print("Not match",idx,index_k,q)
    if abs(np.linalg.norm(bz["k_points"][idx]-k_inbz))>1e-11:
        print("k",bz["k_points"][index_k],"q",q)
        print(tbglib.coeffs(k_inbz),"kplusq",kplusq,tbglib.coeffs(G))
        print(tbglib.in_bz(k_inbz))
        raise ValueError("k plus q does not lie close to any point in bz",
                abs(np.linalg.norm(bz["k_points"][idx]-k_inbz)))
    return idx

if __name__ == "__main__":
    #execution
                
    id = 404
    #solver = hf.hf_solver("data/hf_{}.hdf5".format(id))
    solver = hf.hf_solver("data/coherence/hf_{}.hdf5".format("no_coherence"))
    solver.check_v_c2t_invariance()
    P = solver.P
    bz = solver.bz
    hf_eigenvalues = solver.hf_eigenvalues
    hf_eigenstates = solver.hf_eigenstates
    overlaps = solver.overlaps
    model_params = solver.params

    q = np.array([0,0])
    print("q is equal to:",q)
    k_points = solver.bz["k_points"]
    filling = int(round(np.real(np.trace(P[0]))))
    N = len(k_points)
    H_mode = np.zeros((7*N,7*N),dtype=complex)

    for i in range(8-filling):
        for k in range(N):
            k_plusq = index_kplusq(bz,k,q)
            H_mode[i*N+k,i*N+k] = hf_eigenvalues[k_plusq,i+filling]\
                                    - hf_eigenvalues[k,filling-1]
    # fill in matrix elements
    for l in range(N):
        for k in range(N):
            kplusq = index_kplusq(bz,k,q) # replace by index of k+q in 1st bz
            lplusq = index_kplusq(bz,l,q) #should be l-q? 
            for i in range(8-filling):
                for j in range(8-filling):
                    for g in range(len(bz["G_values"])):
                        #hartree - as p+h attract
                        H_mode[i*N+k,j*N+l]= H_mode[i*N+k,j*N+l]- \
                                    V_matrix_element(g,lplusq,kplusq,
                                        k,l,j+filling,i+filling,filling-1,
                                        filling-1,overlaps,
                                hf_eigenstates,bz)*\
                                model_params["scaling_factor"]**2/\
                                (N*1.5*math.sqrt(3)) +\
                        V_matrix_element(g,lplusq,l,
                                        k,kplusq,
                                        i+filling,filling-1,filling-1,
                                        j+filling,overlaps,
                                hf_eigenstates,bz)*\
                                model_params["scaling_factor"]**2/(N*1.5*math.sqrt(3))
    

    #np.save("data/h_mode_{}.npy".format(id),H_mode)
    np.save("data/coherence/h_mode_{}.npy".format("no_coherence"),H_mode)
    energies, states = np.linalg.eigh(H_mode)
    print("energies:", energies[:5])
    for i in range(8):
        print("EIGSTATES", np.real(hf_eigenstates[0,:,i]))
    print("state:", states[:8-filling,0])
    dfsafdsfs
    for i in range(8):
        if np.abs(hf_eigenstates[0,7,i])**2+np.abs(hf_eigenstates[0,6,i])**2>0.5:
            print(hf_eigenstates[0,:,i])
            print(np.abs(hf_eigenstates[0,7,i])**2+np.abs(hf_eigenstates[0,6,i])**2)

    print("EIGSTATES", hf_eigenstates[0])
    V_matrix_element(1,4,5,4,7,overlaps,
            hf_eigenstates,1,0,0,1,bz)




