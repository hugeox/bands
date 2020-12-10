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
    G_coeffs = bz["G_coeffs"]
    k_points = bz["k_points"]
    lattice, neighbor_table = build_lattice_and_neighbor_table(k_lattice_radius)
    es = []
    ss = []
    s_matrices = []
    for k in k_points:
        energies, states = find_energies(k,
            params = model_params, N_bands = 2, 
            lattice = lattice,
            neighbor_table = neighbor_table,return_states = True )
        #TODO: compute also KE in other valley
        es.append([energies[0],energies[1],energies[0],energies[1]])
        ss.append(states)
    for G in G_coeffs:
        s_matrices.append(shift_matrix(G,lattice))
    
    overlaps = np.zeros((len(G_coeffs),len(k_points),
                        len(k_points),4,4),dtype = complex)
    for i in range(len(k_points)):
        for j in range(len(k_points)):
            for g in range(len(G_coeffs)):
                for m in range(2):
                    for n in range(2):
                        #temporarily ignore valley dependence of overlaps
                        overlaps[g,i,j,m,n] = overlap(ss[i][m],ss[j][n],s_matrices[g])
                        overlaps[g,i,j,m+2,n+2]=overlap(ss[i][m],ss[j][n],s_matrices[g])
    
    return energies, overlaps
def v_hf(bz,overlaps,model_params,P):
    G_coeffs = bz["G_coeffs"]
    G_s = bz["G_values"]
    k_points = bz["k_points"]
    V_coulomb = params["potential"]
    V_hf = np.zeros((len(k_points),4,4),dtype = complex)

    direct_potential = [] #at G
    for g in range(len(G_coeffs)):
        temp = 0
        for k in len(k_points):
            temp = temp + \
                 np.trace(np.matmul(P(k), np.conjugate(overlaps[g,k,k,:,:])))
        direct_potential.append(temp)
    for k in len(k_points):
        temp = np.array((4,4),dtype = complex)
        for g in range(len(G_coeffs)):
            temp = temp + \
                    direct_potential[g]*V_coulomb(G_s[g])*\
                    overlaps[g,k,k,:,:]
            for l in len(k_points):
                temp = temp+\
                    V_coulomb(G_s[g]+k_points[l]-k_points[k])*\
                    np.matmul(np.matmul(overlaps[g,k,l,:,:],
                            np.matrix.transpose(P[l,:,:])),
                            np.matrix.transpose(np.matrix.conjugate(
                            overlaps[g,k,l,:,:])))

        V_hf[k,:,:]=temp
    return V_hf
    
def iterate_hf(bz,energies, overlaps, model_params, P):
    h_mf = np.array(energies) + v_hf(bz,overlaps,model_params, P)
    G_coeffs = bz["G_coeffs"]
    G_s = bz["G_values"]
    k_points = bz["k_points"]
    flavors = len(P[0,:,:])
    P_new = np.zeros(P.shape)
    for k in len(k_points):
        filling = round(np.real(np.trace(P[k,:,:])))
        energies, states = np.linalg.eigh(h_mf[k,:,:])
        m = np.zeros(flavors)
        m[:flavors]=1
        P_new[k,:,:] = np.transpose(states) @\
                    np.diag(m) @ np.conjugate(states)

        
    

if __name__ == "__main__":
    #execution
    ks = np.zeros((20,20))
    for i in range(0,20):
        for j in range(0,20):
            if tbglib.in_bz([i/5-2,j/5-2]):
                ks[i,j]=1
    print(ks)
                
    model_params = {"theta" : 1.09 * np.pi / 180, #twist angle in radians
                    "w_AA" :0,#80 #in meV
                    "w_AB" : 110,#110 #in meV
                    "v_dirac" : int(19746/2), #v_0 k_D in meV
                    "epsilon" : 5
                    }
    k_lattice_radius = 1.2
    lattice, neighbor_table = gbs.build_lattice_and_neighbor_table(
                                    k_lattice_radius)
    N_bands = 2


    k = [0.1,0]
    energies, states= gbs.find_energies(k,
        params = model_params, N_bands = N_bands, 
         lattice = lattice, 
        neighbor_table = neighbor_table, return_states = True)
    print(energies)

    print(len(states))
    print(states)
    print(energies)

    laps = []
    for i in range(1,5):
        mat = shift_matrix([i,-i],lattice)
        laps.append(np.abs(overlap(states[0],states[0],mat)))
        laps.append(np.abs(overlap(states[0],states[1],mat)))
    plt.plot(laps)
    plt.show()

