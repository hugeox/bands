import numpy as np
from math import cos,sin
import math
import time
import matplotlib.pyplot as plt
import generate_band_structure as gbs
import tbglib
import h5py

def V_coulomb(q_vec):
    #returns V(q) in units of meV nm^2
    q=np.linalg.norm(q_vec)
    d_s = 40 #screening lenght in nm
    scaling_factor = 2* sin(1.09*np.pi/180)*\
                4*np.pi/(3*math.sqrt(3)*0.246)
    epsilon=1/0.06 
    if q*scaling_factor*d_s<0.1:
        return 1439.96*d_s*4*np.pi/epsilon #in eV nm^2
    else:
        return 1439.96*2*np.pi*math.tanh(q*scaling_factor*d_s)/\
                (q*scaling_factor*epsilon)

def V_matrix_element(g,k1,k2,k3,k4,i,j,k,l,overlaps,
        hf_eigenstates,bz):
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

if __name__ == "__main__":
    #execution
                
    model_params = {"theta" : 1.09 * np.pi / 180, #twist angle in radians
                    "w_AA" :80, #in meV
                    "w_AB" : 110,#110 #in meV
                    "v_dirac" : int(19746/2), #v_0 k_D in meV
                    "epsilon" : 5,
                    "scaling_factor": 2* sin(1.09*np.pi/180)*\
                    4*np.pi/(3*math.sqrt(3)*0.246) ,
                    "q_lattice_radius": 10,
                    "V_coulomb" : V_coulomb #V_q
                    }

    id = 4
    f_in = h5py.File('hf_newbz{}.hdf5'.format(id), 'r')
    SIZE_BZ = f_in.attrs["size_bz"]
    bz = tbglib.build_bz(SIZE_BZ)
    overlaps = f_in["overlaps"][...]
    sp_energies = f_in["sp_energies"][...]
    P = f_in["P_hf"][...]
    hf_eigenvalues = f_in["hf_eigenvalues"][...]
    hf_eigenstates = f_in["hf_eigenstates"][...]
    V_coulomb_array = f_in["V_coulomb"][...]
    for key in model_params.keys():
        if key !="V_coulomb":
            model_params[key] = f_in.attrs[key] 
    f_in.close()

    q = np.array([0,0])
    #q = bz["k_points_diff"][3]
    print("q is equal to:",q)
    k_points = bz["k_points"]
    filling = int(round(np.real(np.trace(P[0]))))
    N = len(k_points)

    H_mode=np.load("h_mode_{}.npy".format(id))
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




