import numpy as np
from math import cos,sin
import math
import time
import matplotlib.pyplot as plt
import generate_band_structure as gbs
import tbglib
import h5py
import json

def V_coulomb(q_vec):
    #returns V(q) in units of meV nm^2
    q=np.linalg.norm(q_vec)
    d_s = 40 #screening lenght in nm
    scaling_factor = 2* sin(1.09*np.pi/180)*\
                4*np.pi/(3*math.sqrt(3)*0.246)
    epsilon=1/0.06 
    return 0
    if q*scaling_factor*d_s<0.1:
        return 1439.96*d_s*4*np.pi/epsilon #in eV nm^2
    else:
        return 1439.96*2*np.pi*math.tanh(q*scaling_factor*d_s)/\
                (q*scaling_factor*epsilon)

def V_matrix_element(g,k1,k2,k3,k4,overlaps,
        hf_eigenstates,i,j,k,l,bz):
    U = hf_eigenstates
    element = (tbglib.dagger(U[k1]) @ overlaps[g,k1,k2,:,:] @ U[k2])[i,j] *\
                (tbglib.dagger(U[k3]) @ overlaps[bz["G_neg_indices"][g],k3,k4,:,:] @ U[k4])[k,l] *\
                V_coulomb(bz["k_points"][k2]-bz["k_points"][k1]+bz["G_values"][g])
    return element


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

    f_in = h5py.File('hf_1.hdf5', 'r')
    SIZE_BZ = f_in.attrs["size_bz"]
    brillouin_zone = tbglib.build_bz(SIZE_BZ)
    overlaps = f_in["overlaps"].value
    sp_energies = f_in["sp_energies"].value
    P = f_in["P_hf"].value
    hf_eigenvalues = f_in["hf_eigenvalues"].value
    hf_eigenstates = f_in["hf_eigenstates"].value
    V_coulomb_array = f_in["V_coulomb"].value
    for key in model_params.keys():
        if key !="V_coulomb":
            model_params[key] = f_in.attrs[key] 
    f_in.close()
    for i in range(8):
        print("EIGSTATES", np.real(hf_eigenstates[0,:,i]))
    print("EIGSTATES", hf_eigenstates[0])
    V_matrix_element(1,4,5,4,7,overlaps,
            hf_eigenstates,1,0,0,1,brillouin_zone)

    print(P[0])
    print(sp_energies)
    for i in range(4):
        plt.plot(sp_energies[:,i],label = str(i))
        #plt.plot(np.array(energies)[40:60,i])
    plt.legend()
    plt.show()
    for i in range(2):
        plt.plot([np.array(sp_energies)[m,i] for m in brillouin_zone["trajectory"]])

    plt.xticks(brillouin_zone["ticks_coords"],brillouin_zone["ticks_vals"])
    plt.legend()
    plt.show()
    #print(brillouin_zone["k_points"][22:32])



