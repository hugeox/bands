import numpy as np
from math import cos,sin
import math

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

if __name__ == "__main__":

    #tests
    print(build_T_matrices(0,1))
    print(build_T_matrices(1,0))
    print(h_angle([1,0],np.pi/2))
    print(build_lattice(1.5))
    lat = build_lattice(1.5)
    print(build_neighbor_table(lat))

    #execution


    theta = 1.10 * np.pi / 180 #twist angle in radians
    w_AA = 80 #in meV
    w_AB = 110 #in meV
    v_dirac = 13 #including hbar
    k_D = 1 #dirac 
    k_theta = 2 *k_D* sin(theta/2)
    Ts = build_T_matrices(w_AA,w_AB)
    lattice = build_lattice(1.5)
    H = np.zeros((2*len(lattice),2*len(lattice)),dtype=complex)
    print(H)
    neighbor_table = build_neighbor_table(lattice)
    """ fill table with hoppings""" 
    for i,j,n in neighbor_table:
        H[2*i:2*i+2,2*j:2*j+2] = Ts[n]
        H[2*j:2*j+2,2*i:2*i+2] = np.matrix.transpose(np.matrix.conjugate(Ts[n]))#hermitian conjugate of that


    momentum = [0,0]
    """ fill table with kinetic terms""" 
    i = 0
    for lattice_point in lattice:
        k = momentum + lattice_point[0] * tbgHelper.q1 + lattice_point[1] * tbgHelper.q2
        if lattice_point[2] == 0: 
            H[2*i:2*i+2,2*i:2*i+2] =h_angle(k , -theta/2) #bottom layer
        if lattice_point[2] == 1:
            H[2*i:2*i+2,2*i:2*i+2] =h_angle(k , theta/2) #top layer
        i = i+1
    


