import numpy as np
from math import cos,sin
import math
import time
import matplotlib.pyplot as plt
import generate_band_structure as gbs
import tbglib


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
                    energies_prime[0],energies_prime[1]])
        ss.append(states)
        ss_prime.append(np.conjugate(states_prime))
    for G in G_coeffs:
        s_matrices.append(shift_matrix(G,lattice))
        s_matrices_prime.append(shift_matrix(-G,lattice)) #K' fudge
    
    overlaps = np.zeros((len(G_coeffs),len(k_points),
                        len(k_points),4,4),dtype = complex)
    for i in range(len(k_points)):
        for j in range(len(k_points)):
            for g in range(len(G_coeffs)):
                for m in range(2):
                    for n in range(2):
                        overlaps[g,i,j,m,n] = overlap(ss[i][m],ss[j][n],s_matrices[g])
                        overlaps[g,i,j,m+2,n+2]=overlap(ss_prime[i][m],
                                    ss_prime[j][n],s_matrices_prime[g])
    
    return es, overlaps
def v_hf(bz,overlaps,model_params,P):
    G_coeffs = bz["G_coeffs"]
    G_s = bz["G_values"]
    k_points = bz["k_points"]
    V_coulomb = model_params["V_coulomb"]
    V_hf = np.zeros((len(k_points),4,4),dtype = complex)

    direct_potential = [] #at G
    for g in range(len(G_coeffs)):
        temp = 0
        for k in range(len(k_points)):
            temp = temp + \
                 np.trace(np.matmul(P[k], np.conjugate(overlaps[g,k,k,:,:])))
        direct_potential.append(temp)
    for k in range(len(k_points)):
        temp = np.zeros((4,4),dtype = complex)
        for g in range(len(G_coeffs)):
            temp = temp + \
                    np.multiply(direct_potential[g]*V_coulomb(G_s[g]),
                    overlaps[g,k,k,:,:])
            for l in range(len(k_points)):
                #TODO: can be replaced by np.sum
                temp = temp-\
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
    P_new = np.zeros(np.array(P).shape,dtype = complex)
    for k in range(len(k_points)):
        filling = round(np.real(np.trace(P[k])))
        energies, states = np.linalg.eigh(h_mf[k,:,:])
        m = np.zeros(flavors,dtype = complex)
        m[:filling]=1
        P_new[k,:,:] = np.conjugate(states) @\
                     np.diag(m) @  np.transpose(states)
        if k==60 and False:
            print("Energies", energies)
            print("states ", np.real(states))
        hf_energies.append(energies)
    return P_new, hf_energies
    

if __name__ == "__main__":
    #execution
    def V_coulomb(q_vec):
        #returns V(q) in units of meV nm^2
        q=np.linalg.norm(q_vec)
        d_s = 40 #screening lenght in nm
        scaling_factor = 2* sin(1.09*np.pi/180)*\
                    4*np.pi/(3*math.sqrt(3)*0.246)
        # k_real = q*scaling_factor, units in 1/nm
        epsilon=1/0.06 
        if q*scaling_factor*d_s<0.1:
            return 1439.96*d_s*4*np.pi/epsilon #in eV nm^2
        else:
            return 1439.96*2*np.pi*math.tanh(q*scaling_factor*d_s)/\
                    (q*scaling_factor*epsilon)
        
                
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
    brillouin_zone = tbglib.build_bz()
    N_k = len(brillouin_zone["k_points"])
    print("Number of points is:", N_k)
    P_k=np.diag([0,1,0,0])
    P_0 = [P_k for k in range(N_k)]
    #sp_energies, overlaps = build_overlaps(brillouin_zone,model_params)
    #np.save("sp_energies",sp_energies)
    #np.save("overlaps",overlaps)
    sp_energies = np.load("sp_energies.npy")
    overlaps = np.load("overlaps.npy")
    for m in range(1):
        P, energies = iterate_hf(brillouin_zone,sp_energies,overlaps, model_params, P_0)
    for m in range(40):
        P_old = P.copy()
        P, energies = iterate_hf(brillouin_zone,sp_energies,overlaps, model_params, P_old)
        #print("Total hf energy", hf_energy_total(brillouin_zone,sp_energies,overlaps, model_params, P_old))
        print(np.linalg.norm(np.array(P).ravel()-np.array(P_old).ravel()))
    for i in range(4):
        plt.plot(np.array(energies))#[22:32,i])
    plt.show()
    #print(brillouin_zone["k_points"][22:32])


