import numpy as np
from math import cos,sin
import math
import time
import matplotlib.pyplot as plt
import generate_band_structure as gbs
import tbglib
import h5py
import json

""" solve a self-consistent Hartree-Fock at given filling and save into file"""

def V_coulomb(q_vec):
    #returns V(q) in units of meV nm^2
    q=np.linalg.norm(q_vec)
    d_s = 40 #screening lenght in nm
    scaling_factor = 2* sin(1.09*np.pi/180)*\
                4*np.pi/(3*math.sqrt(3)*0.246)
    epsilon=1/0.06
    if q*scaling_factor*d_s<0.01:
        return 1439.96*d_s*4*np.pi/epsilon #in eV nm^2
    else:
        return 1439.96*2*np.pi*math.tanh(q*scaling_factor*d_s)/\
                (q*scaling_factor*epsilon)

def read_hf_from_file(filename):
    hf_solution={}
    model_params = {"theta" : 1.09 * np.pi / 180, #twist angle in radians
                    "w_AA" :80, #in meV
                    "w_AB" : 110,#110 #in meV
                    "v_dirac" : int(19746/2), #v_0 k_D in meV
                    "epsilon" : 1/0.06,
                    "d_s": 40, #screening length in nm
                    "scaling_factor": 2* sin(1.09*np.pi/180)*\
                    4*np.pi/(3*math.sqrt(3)*0.246) ,
                    "single_gate_screening": False, #single or dual gate screening?
                    "q_lattice_radius": 10,
                    "V_coulomb" : V_coulomb #V_q
                    }
    f_in = h5py.File(filename, 'r')
    for k in f_in.keys():
        print(k)
        hf_solution[k] = f_in[k][...]
    SIZE_BZ = f_in.attrs["size_bz"]
    bz = tbglib.build_bz(SIZE_BZ)
    hf_solution["bz"] = bz 
    for key in f_in.attrs.keys():
        if key !="V_coulomb":
            model_params[key] = f_in.attrs[key] 
    f_in.close()
    hf_solution["model_params"] = model_params
    hf_solution["k_points"] = bz["k_points"]
    hf_solution["N"] = len(bz["k_points"])
    return hf_solution

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

def C2T_eigenvalue(state):
    a = int(len(state)/2)
    if a%2==1:
        a = a +1
    eigvalue = np.conjugate(state[a+1])/state[a]
    eigvalue2 = np.conjugate(state[a+3])/state[a+2]
    if abs(abs(eigvalue)-1)>0.00001:
        print("C2T Eigvalue is, and its norm is not one: ", eigvalue)
    if abs(eigvalue-eigvalue2)>0.00001:
        print("C2T Eigvalue is, and is not equal with second method : ", eigvalue)
    return eigvalue

def all_c2t_evals(states,states_prime):
    arr = [C2T_eigenvalue(states[0]),C2T_eigenvalue(states[1]),
        C2T_eigenvalue(states_prime[0]),C2T_eigenvalue(states_prime[1]),
            C2T_eigenvalue(states[0]),C2T_eigenvalue(states[1]),
        C2T_eigenvalue(states_prime[0]),C2T_eigenvalue(states_prime[1])]
    return arr
    
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
    c2t_evals = []
    ss_prime = []
    ss = []
    s_matrices_prime = []
    s_matrices = []
    for k in k_points:
        print(k)
        energies, states = gbs.find_energies(k,
            params = model_params, N_bands = 2, 
            lattice = lattice,
            neighbor_table = neighbor_table,return_states = True )
        print(energies)
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
        c2t_evals.append(all_c2t_evals(states,np.conjugate(states_prime)))
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
    
    return es, overlaps, c2t_evals
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
                 np.trace(P[k] @ np.conjugate(overlaps[g,k,k,:,:]))
        if g==0: #remove 0 component of V
            temp = 0
        direct_potential.append(temp)
    for k in range(len(k_points)):
        temp = np.zeros((8,8),dtype = complex)
        for g in range(len(G_coeffs)):
            temp = temp + \
                    direct_potential[g]*V_coulomb(G_s[g])*\
                    overlaps[g,k,k,:,:]
            for l in range(len(k_points)):
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
        energies, states = np.linalg.eigh(h_mf[k,:,:])
        m = np.zeros(flavors,dtype = complex)
        m[:filling]=1
        P_new[k,:,:] = np.conjugate(states) @\
                     np.diag(m) @  np.transpose(states)
        hf_energies.append(energies)
        hf_states.append(states)
    return P_new, hf_energies, hf_states
    

if __name__ == "__main__":
    #execution

                
    model_params = {"theta" : 1.09 * np.pi / 180, #twist angle in radians
                    "w_AA" :80, #in meV
                    "w_AB" : 110,#110 #in meV
                    "v_dirac" : int(19746/2), #v_0 k_D in meV
                    "epsilon" : 1/0.06,
                    "d_s": 40, #screening length in nm
                    "scaling_factor": 2* sin(1.09*np.pi/180)*\
                    4*np.pi/(3*math.sqrt(3)*0.246) , #this will actually be computed from theta, 0.246nm = lattice const. of graphene
                    "single_gate_screening": False, #single or dual gate screening?
                    "q_lattice_radius": 10,
                    "size_bz": 12,
                    "description": "v=-3, small bz, but new ",
                    "V_coulomb" : V_coulomb,
                    "filling": 1
                    }
    brillouin_zone = tbglib.build_bz(model_params["size_bz"])
    N_k = len(brillouin_zone["k_points"])
    print("Number of points is:", N_k)
    a = np.array([1,0,0,0,0,0,0,0])
    a[:model_params["filling"]] = 1
    print(a)
    P_k=np.diag(a)
    P_0 = [P_k for k in range(N_k)]
    id = 1
    print(id)
    sp_energies, overlaps, c2t_eigenvalues = build_overlaps(brillouin_zone,model_params)

    for m in range(1):
        P, energies,states = iterate_hf(brillouin_zone,sp_energies,overlaps, model_params, P_0)
    for m in range(50):
        P_old = P.copy()
        P, hf_eig,hf_states = iterate_hf(brillouin_zone,sp_energies,overlaps, model_params, P_old)
        #print("Total hf energy", hf_energy_total(brillouin_zone,sp_energies,overlaps, model_params, P_old))
        print(np.linalg.norm(np.array(P).ravel()-np.array(P_old).ravel()))
    while input("continue? y/n") == "y":
        for m in range(50):
            P_old = P.copy()
            P, hf_eig,hf_states = iterate_hf(brillouin_zone,sp_energies,overlaps, model_params, P_old)
            #print("Total hf energy", hf_energy_total(brillouin_zone,sp_energies,overlaps, model_params, P_old))
            print(np.linalg.norm(np.array(P).ravel()-np.array(P_old).ravel()))
    
    f_out = h5py.File('data/hf_{}.hdf5'.format(id), 'w')
    f_out.create_dataset("overlaps", data = overlaps)
    f_out.create_dataset("sp_energies", data = sp_energies)
    f_out.create_dataset("P_hf", data = P)
    f_out.create_dataset("P_0", data = P_0)#save also inital guess
    f_out.create_dataset("hf_eigenvalues", data = hf_eig)
    f_out.create_dataset("hf_eigenstates", data = hf_states)
    f_out.create_dataset("c2t_eigenvalues", data = c2t_eigenvalues)
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



