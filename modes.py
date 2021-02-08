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
        hf_eigenstates,bz,V_coulomb):
    U = hf_eigenstates
    element = (tbglib.dagger(U[k1]) @ overlaps[g,k1,k2,:,:] @ U[k2])[i,j] *\
                (tbglib.dagger(U[k3]) @ overlaps[bz["G_neg_indices"][g],k3,k4,:,:] @ U[k4])[k,l] *\
                V_coulomb(bz["k_points"][k2]-bz["k_points"][k1]+bz["G_values"][g])
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
def mode_solver(object):
    def __init__(self, q, hf_solver,h_mode_filename = None):
        self.q = q
        if type(hf_solver)==string:
            self.solver = hf.hf_solver(hf_solver)
        else:
            self.solver =  hf_solver
        self.N_f = solver.params["N_f"]
        self.full_fill = int(N_f/2) #filling when totally full
        self.N_filled = solver.params["filling"] + full_fill
        self.N_empty = N_f - N_filled
        k_points = solver.bz["k_points"]
        self.N = len(k_points)
        if h_mode_filename is None:
            self.H_mode = self.build_H_mode()
        else:
            self.H_mode = np.load(h_mode_filename)
    def build_H_mode(self):
        q = self.q
        P = self.solver.P
        bz = self.solver.bz
        hf_eigenvalues = self.solver.hf_eigenvalues
        hf_eigenstates = self.solver.hf_eigenstates
        overlaps = self.solver.overlaps
        model_params = self.solver.params
        k_points = solver.bz["k_points"]
        N_f = solver.params["N_f"]
        full_fill = int(N_f/2) #filling when totally full
        N_filled = solver.params["filling"] + full_fill
        N_empty = N_f - N_filled
        N = len(k_points)
        H_mode = np.zeros((N,N_filled,N_empty,N,N_filled,N_empty),dtype=complex)
        for i in range(N_filled):
            for j in range(N_empty):
                for k in range(N):
                    k_plusq = index_kplusq(bz,k,q)
                    H_mode[k,i,j,k,i,j] = hf_eigenvalues[k_plusq,j+N_filled]\
                                            - hf_eigenvalues[k,i]
        for k in range(N):
            for l in range(N):
                kplusq = index_kplusq(bz,k,q) # replace by index of k+q in 1st bz
                lplusq = index_kplusq(bz,l,q) #should be l-q? 
                for filled_l in range(N_filled):
                    for filled_r in range(N_filled):
                        for empty_l in range(N_empty):
                            for empty_r in range(N_empty):
                                for g in range(len(bz["G_values"])):
                                    H_mode[k,filled_l,empty_l,l, filled_r,empty_r]= \
                                    H_mode[k,filled_l,empty_l,l,filled_r,empty_r]-\
                                                V_matrix_element(g,lplusq,kplusq,
                                                    k,l,empty_r+N_filled,empty_l+N_filled,filled_l,
                                                    filled_r,overlaps,
                                            hf_eigenstates,bz,solver.V_coulomb)*\
                                            model_params["scaling_factor"]**2/\
                                            (N*1.5*math.sqrt(3)) +\
                                        V_matrix_element(g,lplusq,l,
                                                        k,kplusq,
                                                        empty_r+N_filled,filled_r,filled_l,
                                                        empty_l+N_filled,overlaps,
                                                hf_eigenstates,bz,solver.V_coulomb)*\
                                                model_params["scaling_factor"]**2/(N*1.5*math.sqrt(3))
        self.H_mode = H_mode
    def solve(self,N_states):
        energies, states = np.linalg.eigh(np.reshape(self.H_mode,(self.N*self.N_filled*self.N_empty,
                                    self.N*self.N_filled*self.N_empty)))
        
        states_new = [ np.reshape(states[:,i],(N,N_filled,N_empty)) for i in range(N_states)]
        return energies[:N_states], states_new
            
if __name__ == "__main__":
    #execution
                
    id =   5
    solver = hf.hf_solver("data/hf_{}.hdf5".format(id))
    #solver = hf.hf_solver("data/coherence/hf_{}.hdf5".format("no_coherence"))
    print(solver.params)
    P = solver.P
    bz = solver.bz
    hf_eigenvalues = solver.hf_eigenvalues
    hf_eigenstates = solver.hf_eigenstates
    overlaps = solver.overlaps
    model_params = solver.params

    q = np.array([0,0])
    print("q is equal to:",q)
    k_points = solver.bz["k_points"]
    N_f = solver.params["N_f"]
    full_fill = int(N_f/2) #filling when totally full
    N_filled = solver.params["filling"] + full_fill
    N_empty = N_f - N_filled
    N = len(k_points)
    H_mode = np.zeros((N,N_filled,N_empty,N,N_filled,N_empty),dtype=complex)
    print(H_mode.shape)

    """
    H_mode = np.load("data/h_mode_{}.npy".format(id))
    energies, states = np.linalg.eigh(np.reshape(H_mode,(N*N_filled*N_empty,
                                N*N_filled*N_empty)))
    print("energies:", energies[:5])
    state = np.reshape(states[:,0],(N,N_filled,N_empty))
    print(state[0,0,:])
    print("First eigenstate", np.abs(hf_eigenstates[0][:,0]))
    print(np.abs(state[0,0,4]*hf_eigenstates[0][:,5] +
state[0,0,6]*hf_eigenstates[0][:,7]))
    print(hf_eigenstates[0][:,7])
    print(hf_eigenstates[0][:,5])
    state = np.reshape(states[:,1],(N,N_filled,N_empty))
    print(hf_eigenstates[0][:,1])
    print(hf_eigenstates[0][:,3])
    print(state[0,0,:])
    sfd
    """ 
    for i in range(N_filled):
        for j in range(N_empty):
            for k in range(N):
                k_plusq = index_kplusq(bz,k,q)
                H_mode[k,i,j,k,i,j] = hf_eigenvalues[k_plusq,j+N_filled]\
                                        - hf_eigenvalues[k,i]
    for k in range(N):
        for l in range(N):
            kplusq = index_kplusq(bz,k,q) # replace by index of k+q in 1st bz
            lplusq = index_kplusq(bz,l,q) #should be l-q? 
            for filled_l in range(N_filled):
                for filled_r in range(N_filled):
                    for empty_l in range(N_empty):
                        for empty_r in range(N_empty):
                            for g in range(len(bz["G_values"])):
                                H_mode[k,filled_l,empty_l,l, filled_r,empty_r]= \
                                H_mode[k,filled_l,empty_l,l,filled_r,empty_r]-\
                                            V_matrix_element(g,lplusq,kplusq,
                                                k,l,empty_r+N_filled,empty_l+N_filled,filled_l,
                                                filled_r,overlaps,
                                        hf_eigenstates,bz,solver.V_coulomb)*\
                                        model_params["scaling_factor"]**2/\
                                        (N*1.5*math.sqrt(3)) +\
                                    V_matrix_element(g,lplusq,l,
                                                    k,kplusq,
                                                    empty_r+N_filled,filled_r,filled_l,
                                                    empty_l+N_filled,overlaps,
                                            hf_eigenstates,bz,solver.V_coulomb)*\
                                            model_params["scaling_factor"]**2/(N*1.5*math.sqrt(3))
    #np.save("data/h_mode_{}.npy".format(),H_mode)
    np.save("data/coherence/h_mode_{}.npy".format("no_coherence"),H_mode)
    energies, states = np.linalg.eigh(np.reshape(H_mode,(N*N_filled*N_empty,
                                N*N_filled*N_empty)))
    print("energies:", energies[:10])
    state = np.reshape(states[:,0],(N,N_filled,N_empty))
    print(state[0,0,:])
    # fill in matrix elements
    

