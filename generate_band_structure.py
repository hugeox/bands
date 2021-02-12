import numpy as np
from math import cos,sin
import math
import time
import matplotlib.pyplot as plt
import tbglib

s0 = np.array([[1, 0],[ 0, 1]])
sx = np.array([[0, 1],[ 1, 0]])
sy = np.array([[0, -1j],[1j, 0]])
sz = np.array([[1, 0],[0, -1]])


def build_lattice(radius):
    size = math.ceil(radius)
    lattice = [] 
    # 0,0 coordinate in terms of q1, q2 in layer 0
    if tbglib.norm(tbglib.coords([0,0])) < radius:
        lattice.append([0,0,0])
    if tbglib.norm(tbglib.coords([1,0])) < radius:
        lattice.append([1,0,1])
    for i in range(-size,size):
        for j in range(-size,size):
            if i==j==0:
                continue
            x = i*tbglib.g1_coeff[0] + j*tbglib.g2_coeff[0]
            y = i*tbglib.g1_coeff[1] + j*tbglib.g2_coeff[1]
            if tbglib.norm(tbglib.coords([x,y])) < radius:
                lattice.append([x,y,0])
            if tbglib.norm(tbglib.coords([x+1,y])) < radius:
                lattice.append([x+1,y,1])
    # layer symmetric way of doing it - for each layer consider
    # k + G where |G|<radius, preserved under T
    # for i in range(-size,size):
    #    for j in range(-size,size):
    #        x = i*tbglib.g1_coeff[0] + j*tbglib.g2_coeff[0]
    #        y = i*tbglib.g1_coeff[1] + j*tbglib.g2_coeff[1]
    #        if tbglib.norm([x,y]) < radius:
    #            lattice.append([x,y,0])
    #            lattice.append([x+1,y,1])
    return lattice

def build_neighbor_table(lattice):
    """ format [first,second, direction]"""
    neighbors = []
    size = len(lattice)
    for i in range(-size,size):
        for j in range(-size,size):
            x = i*tbglib.g1_coeff[0] + j*tbglib.g2_coeff[0]
            y = i*tbglib.g1_coeff[1] + j*tbglib.g2_coeff[1]
            try:
                index1 = lattice.index([x,y,0])
                index2 = lattice.index([x+1,y,1])
                neighbors.append([index1,index2,0])
            except:
                2+3
            try:
                index1 = lattice.index([x,y,0])
                index2 = lattice.index([x,y+1,1])
                neighbors.append([index1,index2,1])
            except:
                2+3
            try:
                index1 = lattice.index([x,y,0])
                index2 = lattice.index([x-1,y-1,1])
                neighbors.append([index1,index2,2])
            except:
                2+3
    return neighbors
def build_lattice_and_neighbor_table(radius):
    
    lattice = build_lattice(radius)
    neighbor_table = build_neighbor_table(lattice)
    return lattice, neighbor_table

def h_angle(k,theta):
    #theta = 0 #gives Particle-Hole symmetry
    k_rot = np.dot(np.array([[cos(theta),sin(theta)],
        [-sin(theta),cos(theta)]]),
        k)
    return k_rot[0]*tbglib.sx + k_rot[1]*tbglib.sy

def build_T_matrices(w_AA,w_AB,strain):
    Ts = []
    for k in range(3):
        if k ==0:
            Ts.append((w_AA*s0 
                    + w_AB*cos(k*tbglib.phi)*tbglib.sx
                    + w_AB*sin(k*tbglib.phi)*tbglib.sy)*(1+strain))
        else:
            Ts.append(w_AA*s0 
                    + w_AB*cos(k*tbglib.phi)*tbglib.sx
                    + w_AB*sin(k*tbglib.phi)*tbglib.sy)
    return Ts


def find_energies(k_eval, params, N_bands ,k_lattice_radius=10.5, lattice = None, neighbor_table = None,
                    return_states = False):
    v_dirac= params["v_dirac"]
    theta=params["theta"]
    ph = params["ph"]
    strain = params["strain"]
    Ts = build_T_matrices(params["w_AA"],params["w_AB"],strain)
    H = np.zeros((2*len(lattice),2*len(lattice)), dtype=complex)
    """ fill table with hoppings""" 	
    st = time.time()
    for i,j,n in neighbor_table:
        H[2*i:2*i+2,2*j:2*j+2] = Ts[n]
        H[2*j:2*j+2,2*i:2*i+2] = np.matrix.transpose(np.matrix.conjugate(Ts[n]))#hermitian conjugate of that
    end = time.time()
    #print("Build hopping table in", end-st)
    st = time.time()
    """ fill table with kinetic terms""" 
    i = 0
    for lattice_point in lattice:
        k = k_eval + lattice_point[0] * tbglib.q1 + lattice_point[1] * tbglib.q2
        if lattice_point[2] == 0: 
            if ph:
                H[2*i:2*i+2,2*i:2*i+2] = 2 * sin(theta/2) *v_dirac *h_angle(k ,0) #bottom layer
            else:
                H[2*i:2*i+2,2*i:2*i+2] = 2 * sin(theta/2) *v_dirac *h_angle(k , -theta/2) #bottom layer
        if lattice_point[2] == 1:
            if ph:
                H[2*i:2*i+2,2*i:2*i+2] =2 * sin(theta/2) *v_dirac*h_angle(k , 0) #top layer
            else:
                H[2*i:2*i+2,2*i:2*i+2] =2 * sin(theta/2) *v_dirac*h_angle(k , theta/2) #top layer
        i = i+1
    end = time.time()

    if return_states:
        energies, states = np.linalg.eigh(H)
        zipped = sorted(sorted(zip(energies,np.transpose(states)),key=lambda pair:abs(pair[0]))[:N_bands],key=lambda pair:pair[0])
        tuples = zip(*zipped)
        energies, states = [list(t) for t in tuples]
        return energies,np.array(states)
    else:
        energies = sorted(np.linalg.eigvalsh(H),key = abs)
        return sorted(energies[:N_bands])

if __name__ == "__main__":
    #execution

    model_params = {"theta" : 1.09 * np.pi / 180, #twist angle in radians
                    "w_AA" :80,#80 #in meV
                    "w_AB" : 110,#110 #in meV
                    "v_dirac" : int(19746/2), #v_0 k_D in meV
                    "epsilon" : 5
                    }
    k_lattice_radius =  5
    lattice, neighbor_table = build_lattice_and_neighbor_table(k_lattice_radius)
    N_bands = 2
    bz = tbglib.build_bz(60)


    if True:
        bands_transposed = []
        
        for k in bz["k_points"]:
            energies, states= find_energies(k,
                params = model_params, N_bands = N_bands, 
                lattice = lattice,
                neighbor_table = neighbor_table,return_states = True )
            bands_transposed.append(energies)
        bands = np.transpose(bands_transposed)

        print(states[0])
        #print(len(states[0]))
        print(len(lattice))
        for i in range(N_bands):
            plt.plot([np.array(bands_transposed)[m,i] for m in bz["trajectory"]])

        plt.xticks(bz["ticks_coords"],bz["ticks_vals"])
        plt.legend()
        plt.show()
        print(energies)
        plt.show()

    if False:
        to_plot = np.zeros([20,20])
        for m in range(20):
            for n in range(20):
                energies, states= find_energies([m/10,n/10],
                    w_AA, w_AB, v_dirac, N_bands = N_bands, theta = theta,
                    k_lattice_radius=k_lattice_radius, lattice = lattice, 
                    neighbor_table = neighbor_table, return_states = True)
                energies2, states2= find_energies([m/10+0.3,n/10+0.3],
                    w_AA, w_AB, v_dirac, N_bands = N_bands, theta = theta,
                    k_lattice_radius=k_lattice_radius, lattice = lattice, 
                    neighbor_table = neighbor_table, return_states = True)
                to_plot[m,n]= abs(np.dot(states[0],states2[0]))
        plt.plot(to_plot[0])
        plt.show()
