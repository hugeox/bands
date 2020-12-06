import numpy as np
from math import cos,sin
import math
import time
import matplotlib.pyplot as plt

s0 = np.array([[1, 0],[ 0, 1]])
sx = np.array([[0, 1],[ 1, 0]])
sy = np.array([[0, -1j],[1j, 0]])
sz = np.array([[1, 0],[0, -1]])

class tbgHelper:
    q1 = np.array([0, -1.0])
    q2 = np.array([math.sqrt(3)/2,1.0/2])
    q3 = np.array([-math.sqrt(3)/2,1.0/2])
    qs = [q1,q2,q3]
    g1 = [1,-1] #coords in terms of q1,q2
    g2 = [1,2]
    phi = 2 * np.pi /3
    def norm(x,y):
        """ takes integer indices in teerms of q1, q2 returns norm"""
        return np.linalg.norm(x*tbgHelper.q1+y*tbgHelper.q2)


def build_lattice(radius):
    #assert type(size)=Int
    size = math.ceil(radius)
    lattice = [] 
    # 0,0 coordinate in terms of q1, q2 in layer 0
    for i in range(-size,size):
        for j in range(-size,size):
            x = i*tbgHelper.g1[0] + j*tbgHelper.g2[0]
            y = i*tbgHelper.g1[1] + j*tbgHelper.g2[1]
            if tbgHelper.norm(x,y) < radius:
                lattice.append([x,y,0])
            if tbgHelper.norm(x+1,y) < radius:
                lattice.append([x+1,y,1])
    return lattice

def build_neighbor_table(lattice):
    """ format [first,second, direction]"""
    neighbors = []
    size = len(lattice)
    for i in range(-size,size):
        for j in range(-size,size):
            x = i*tbgHelper.g1[0] + j*tbgHelper.g2[0]
            y = i*tbgHelper.g1[1] + j*tbgHelper.g2[1]
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
    k_rot = np.dot(np.array([[cos(theta),sin(theta)],
        [-sin(theta),cos(theta)]]),
        k)
    return k_rot[0]*sx + k_rot[1]*sy

def build_T_matrices(w_AA,w_AB):
    Ts = []
    for k in range(3):
        Ts.append(w_AA*s0 
                + w_AB*cos(k*tbgHelper.phi)*sx
                + w_AB*sin(k*tbgHelper.phi)*sy)
    return Ts


def find_energies(k_eval,w_AA, w_AB, v_dirac, N_bands, theta ,k_lattice_radius=10.5, lattice = None, neighbor_table = None, return_states = False):
    if lattice == None:
        lattice, neighbor_table = build_lattice_and_neighbor_table(k_lattice_radius)
    Ts = build_T_matrices(w_AA,w_AB)
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
        k = k_eval + lattice_point[0] * tbgHelper.q1 + lattice_point[1] * tbgHelper.q2
        if lattice_point[2] == 0: 
            H[2*i:2*i+2,2*i:2*i+2] = 2 * sin(theta/2) *v_dirac *h_angle(k , -theta/2) #bottom layer
        if lattice_point[2] == 1:
            H[2*i:2*i+2,2*i:2*i+2] =2 * sin(theta/2) *v_dirac*h_angle(k , theta/2) #top layer
        i = i+1
    end = time.time()

    if return_states:
        energies, states = np.linalg.eigh(H)
        zipped = sorted(sorted(zip(energies,states),key=lambda pair:abs(pair[0]))[:N_bands])
        tuples = zip(*zipped)
        energies, states = [list(t) for t in tuples]
        return energies,states
    else:
        energies = sorted(np.linalg.eigvalsh(H),key = abs)
        return sorted(energies[:N_bands])

if __name__ == "__main__":
    #execution

    theta = 1.09 * np.pi / 180 #twist angle in radians
    #theta = 5 * np.pi / 180 #twist angle in radians
    w_AA =80 #in meV
    w_AB = 110#110 #in meV
    v_dirac = int(19746/2) #v_0 k_D in meV
    k_lattice_radius =  5
    lattice, neighbor_table = build_lattice_and_neighbor_table(k_lattice_radius)
    print(sin(theta/2)*v_dirac)
    N_bands = 2

    k_points_to_eval =[]
    for m in range(-10,10):
        k_points_to_eval.append(-m*tbgHelper.q1/9)
    for m in range(1,20):
        k_points_to_eval.append(-tbgHelper.q1+ (-tbgHelper.q3+2*tbgHelper.q1)*m/19)
    for m in range(1,10):
        k_points_to_eval.append(-tbgHelper.q3-tbgHelper.q2*m/9)


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
    if True:
        bands_transposed = []
        
        for k in k_points_to_eval:
            energies, states = find_energies(k,
                w_AA, w_AB, v_dirac, N_bands = N_bands, theta = theta,
                k_lattice_radius=k_lattice_radius, lattice = lattice, neighbor_table = neighbor_table,return_states = True )
            bands_transposed.append(energies)
        bands = np.transpose(bands_transposed)

        for band in bands:
            plt.plot(band)
        print(energies)
        plt.show()

