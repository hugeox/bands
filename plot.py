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

    f_in = h5py.File('hf_newbz5.hdf5', 'r')
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

 
    k_points = bz["k_points"]
    N = len(k_points)

    """ PLOTTING """

    for i in range(8):
        print(hf_eigenstates[0,:,i])
        plt.plot([np.array(hf_eigenvalues)[m,i] for m in bz["trajectory"]])
    plt.xticks(bz["ticks_coords"],bz["ticks_vals"])
    plt.legend()
    plt.show()

