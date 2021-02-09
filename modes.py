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
class mode_solver(hf.hf_solver):
    def build_H_mode(self,q):
        print("Building H MODE")
        N_f = self.params["N_f"]
        full_fill = int(N_f/2) #filling when totally full
        N_filled = self.params["filling"] + full_fill
        N_empty = N_f - N_filled
        P = self.P
        bz = self.bz
        hf_eigenvalues = self.hf_eigenvalues
        hf_eigenstates = self.hf_eigenstates
        overlaps = self.overlaps
        model_params = self.params
        k_points = self.bz["k_points"]
        N = self.N_k
        H_mode = np.zeros((N,N_filled,N_empty,N,N_filled,N_empty),dtype=complex)
        print(H_mode.shape)
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
                                            hf_eigenstates,bz,self.V_coulomb)*\
                                            model_params["scaling_factor"]**2/\
                                            (N*1.5*math.sqrt(3)) +\
                                        V_matrix_element(g,lplusq,l,
                                                        k,kplusq,
                                                        empty_r+N_filled,filled_r,filled_l,
                                                        empty_l+N_filled,overlaps,
                                                hf_eigenstates,bz,self.V_coulomb)*\
                                                model_params["scaling_factor"]**2/(N*1.5*math.sqrt(3))
        self.H_mode = H_mode
    def solve(self,N_states):
        N_f = self.params["N_f"]
        full_fill = int(N_f/2) #filling when totally full
        N_filled = self.params["filling"] + full_fill
        N_empty = N_f - N_filled
        energies, states = np.linalg.eigh(np.reshape(self.H_mode,(self.N_k*N_filled*N_empty,
                                    self.N_k*N_filled*N_empty)))
        
        states_new = [ np.reshape(states[:,i],(self.N_k,N_filled,N_empty)) for i in range(N_states)]
        return energies[:N_states], states_new
            
if __name__ == "__main__":
    #execution
                
    id =   5
    q = np.array([0,0])
    filename = "data/hf_5.hdf5"
    filename = "data/coherence/hf_no_coherence.hdf5"
    solver = mode_solver(filename)
    solver.build_H_mode(q)
    filename = "data/coherence/hf_no_coherence.hdf5"
    solver.save(filename)
    energies,states = solver.solve(10)

    print("energies:", energies)
    print(states[0][0,0,:])
    #solver.save("data/h_mode_{}.npy".format(id))
    solver.save("data/coherence/h_mode_{}.npy".format("no_coherence"))
    print("First eigenstate", np.abs(solver.solver.hf_eigenstates[0][:,0]))
    print(np.abs(state[0,0,4]*solver.solver.hf_eigenstates[0][:,5] +
state[0,0,6]*hf_eigenstates[0][:,7]))
    print(hf_eigenstates[0][:,7])
    print(hf_eigenstates[0][:,5])
    print(hf_eigenstates[0][:,1])
    print(hf_eigenstates[0][:,3])
    print(states[0][0,0,:])
    sfd
    

