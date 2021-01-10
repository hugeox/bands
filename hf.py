import numpy as np
from math import cos,sin
import math
import time
import matplotlib.pyplot as plt
import generate_band_structure as gbs
import tbglib
import h5py
import json


def shift_matrix(G,lattice):
    """returns matrix that shifts components of spinor by G, G written in terms of q1,q2 When some lattice points after shift lie outside range, ignore. Same with points which lie originally outside = set them to 0"""
    M = np.zeros((2*len(lattice),2*len(lattice)))
    i=0
    for x,y,layer in lattice:
        try:
            index1 = lattice.index([x+G[0],y+G[1],layer])
            M[2*index1,2*i]=1
            M[2*index1+1,2*i+1]=1
        except:
            2+3
        i=i+1
    return M
def overlap(state1,state2,shift_matrix):
    """dot product <state1| e^iGr|state2>"""
    return np.vdot(state1,np.dot(shift_matrix,state2))
def build_overlaps(bz,model_params):
    """ singleparticle energies and overlaps"""
    G_coeffs = bz["G_coeffs"]
    k_points = bz["k_points"]
    q_lattice_radius = model_params["q_lattice_radius"]
    lattice, neighbor_table = gbs.build_lattice_and_neighbor_table(q_lattice_radius)
    es = []
    ss_prime = []
    ss = []
    s_matrices_prime = []
    s_matrices = []
    for k in k_points:
        energies, states = gbs.find_energies(k,
            params = model_params, N_bands = 2, 
            lattice = lattice,
            neighbor_table = neighbor_table,return_states = True )
        # fudge K' using time reversal
        energies_prime, states_prime = gbs.find_energies(-np.array(k),
            params = model_params, N_bands = 2, 
            lattice = lattice,
            neighbor_table = neighbor_table,return_states = True )
        es.append([energies[0],energies[1],
                    energies_prime[0],energies_prime[1],
                    energies[0],energies[1],
                    energies_prime[0],energies_prime[1]])
        ss.append(states)
        ss_prime.append(np.conjugate(states_prime))
    for G in G_coeffs:
        s_matrices.append(shift_matrix(G,lattice))
        s_matrices_prime.append(shift_matrix(-G,lattice)) #K' fudge
    
    overlaps = np.zeros((len(G_coeffs),len(k_points),
                        len(k_points),8,8),dtype = complex)
    for i in range(len(k_points)):
        print(i)
        for j in range(len(k_points)):
            for g in range(len(G_coeffs)):
                for m in range(2):
                    for n in range(2):
                        overlaps[g,i,j,m,n] = overlap(ss[i][m],ss[j][n],s_matrices[g])
                        overlaps[g,i,j,m+2,n+2]=overlap(ss_prime[i][m],
                                    ss_prime[j][n],s_matrices_prime[g])
                        #second spin component
                        overlaps[g,i,j,m+4,n+4] = overlaps[g,i,j,m,n] 
                        overlaps[g,i,j,m+6,n+6]= overlaps[g,i,j,m+2,n+2]
    
    return es, overlaps
def v_hf(bz,overlaps,model_params,P):
    G_coeffs = bz["G_coeffs"]
    G_s = bz["G_values"]
    k_points = bz["k_points"]
    V_coulomb = model_params["V_coulomb"]
    V_hf = np.zeros((len(k_points),8,8),dtype = complex)

    direct_potential = [] #at G
    for g in range(len(G_coeffs)):
        temp = 0
        for k in range(len(k_points)):
            temp = temp + \
                 np.trace(np.matmul(P[k], np.conjugate(overlaps[g,k,k,:,:])))
        direct_potential.append(temp)
    for k in range(len(k_points)):
        temp = np.zeros((8,8),dtype = complex)
        for g in range(len(G_coeffs)):
            temp = temp + \
                    np.multiply(direct_potential[g]*V_coulomb(G_s[g]),
                    overlaps[g,k,k,:,:])
            for l in range(len(k_points)):
                #TODO: can be replaced by np.sum
                temp = temp -\
                    V_coulomb(G_s[g]+k_points[l]-k_points[k])*\
                    overlaps[g,k,l,:,:]@\
                            np.matrix.transpose(P[l])@\
                            np.matrix.transpose(np.matrix.conjugate(
                            overlaps[g,k,l,:,:]))


        V_hf[k,:,:]=model_params["scaling_factor"]**2*temp/(len(k_points)*1.5*math.sqrt(3))
                        #area of hexagon
    return V_hf

def hf_energy_total(bz,energies, overlaps, model_params, P):
    temp = 0
    v_mf = v_hf(bz,overlaps,model_params, P)
    en = []
    for e in energies:
        en.append(np.diag(e))
    en = np.array(en) 
    k_points = bz["k_points"]
    for k in range(len(k_points)):
        temp = temp + np.trace(np.transpose(P[k]) @ (en[k] + 0.5*v_mf[k]))
    return temp
    
def iterate_hf(bz,energies, overlaps, model_params, P):
    en = []
    for e in energies:
        en.append(np.diag(e))
    h_mf = np.array(en) + v_hf(bz,overlaps,model_params, P)
    G_coeffs = bz["G_coeffs"]
    G_s = bz["G_values"]
    k_points = bz["k_points"]
    flavors = len(P[0])
    hf_energies = []
    hf_states = []
    P_new = np.zeros(np.array(P).shape,dtype = complex)
    for k in range(len(k_points)):
        filling = int(round(np.real(np.trace(P[k]))))
        #project H_hf onto top 4 bands:
        projector = np.diag([0+0j,1+0j,0,1,0,1,0,1])
        energies, states = np.linalg.eigh(projector@ h_mf[k,:,:] @ projector)
        m = np.zeros(flavors,dtype = complex)
        m[:filling]=1
        P_new[k,:,:] = np.conjugate(states) @\
                     np.diag(m) @  np.transpose(states)
        if k==60 and False:
            print("Energies", energies)
            print("states ", np.real(states))
        hf_energies.append(energies)
        hf_states.append(states)
    return P_new, hf_energies, hf_states
    

if __name__ == "__main__":
    #execution
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

    def build_V_coulomb_table(bz,V_coulomb):
        G_s = bz["G_values"]
        k_points = bz["k_points"]
        coulomb_table = np.ones((len(G_s),len(k_points),len(k_points)))
        for k in range(len(k_points)):
            for l in range(len(k_points)):
                for g in range(len(G_s)):
                    coulomb_table[g,k,l] = V_coulomb(k+g-l)
        return coulomb_table
        
                
    model_params = {"theta" : 1.09 * np.pi / 180, #twist angle in radians
                    "w_AA" :80, #in meV
                    "w_AB" : 110,#110 #in meV
                    "v_dirac" : int(19746/2), #v_0 k_D in meV
                    "epsilon" : 5,
                    "scaling_factor": 2* sin(1.09*np.pi/180)*\
                    4*np.pi/(3*math.sqrt(3)*0.246) ,
                    "q_lattice_radius": 10,
                    "V_coulomb" : V_coulomb, #V_q
                    "size_bz" : 10
                    }
    brillouin_zone = tbglib.build_bz(model_params["size_bz"])
    N_k = len(brillouin_zone["k_points"])
    print("Number of points is:", N_k)
    P_k=np.diag([1,1,1,1,1,1,1,0])
    P_0 = [P_k for k in range(N_k)]
    id = 7
    print(id)
    sp_energies, overlaps = build_overlaps(brillouin_zone,model_params)

    for m in range(1):
        P, energies,states = iterate_hf(brillouin_zone,sp_energies,overlaps, model_params, P_0)
    for m in range(20):
        P_old = P.copy()
        P, hf_eig,hf_states = iterate_hf(brillouin_zone,sp_energies,overlaps, model_params, P_old)
        #print("Total hf energy", hf_energy_total(brillouin_zone,sp_energies,overlaps, model_params, P_old))
        print(np.linalg.norm(np.array(P).ravel()-np.array(P_old).ravel()))
    
    f_out = h5py.File('hf_{}.hdf5'.format(id), 'w')
    f_out.create_dataset("overlaps", data = overlaps)
    f_out.create_dataset("sp_energies", data = sp_energies)
    f_out.create_dataset("P_hf", data = P)
    f_out.create_dtaset("hf_eigenvalues", data = hf_eig)
    f_out.create_dataset("hf_eigenstates", data = hf_states)
    f_out.create_dataset("V_coulomb", data =
            build_V_coulomb_table(brillouin_zone,model_params["V_coulomb"]))
    #f_out.create_group("bz")
    #f_out.attrs["size_bz"] = SIZE_BZ
    #for key in brillouin_zone.keys():
    #    if type(brillouin_zone[key])==dict:
    #        f_out["bz"].create_dataset(key, data = str(brillouin_zone[key]))
    #    else:
    #        f_out["bz"].create_dataset(key, data = brillouin_zone[key])
    for key in model_params.keys():
        if key !="V_coulomb":
            f_out.attrs[key] = model_params[key]
    f_out.close()
    print(P[0])
    print(type(sp_energies))
    for i in range(4):
        plt.plot(np.array(sp_energies)[30:60,i],label = str(i))
        #plt.plot(np.array(energies)[40:60,i])
    plt.legend()
    plt.show()
    for i in range(4):
        plt.plot([np.array(hf_eig)[m,i] for m in brillouin_zone["trajectory"]])

    plt.xticks(brillouin_zone["ticks_coords"],brillouin_zone["ticks_vals"])
    plt.legend()
    plt.show()
    #print(brillouin_zone["k_points"][22:32])



