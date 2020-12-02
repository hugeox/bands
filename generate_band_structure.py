import numpy as np
import math

class tbgHelper:
    q1 = np.array([0, -1.0])
    q2 = np.array([math.sqrt(3)/2,1.0/2])
    q3 = np.array([-math.sqrt(3)/2,1.0/2])
    qs = [q1,q2,q3]
    g1 = [1,-1] #coords in terms of q1,q2
    g2 = [1,2]
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
    neighbors = []
    size = len(lattice)
    for i in range(-size,size):
        for j in range(-size,size):
            x = i*tbgHelper.g1[0] + j*tbgHelper.g2[0]
            y = i*tbgHelper.g1[1] + j*tbgHelper.g2[1]
            try:
                index1 = lattice.index([x,y,0])
                index2 = lattice.index([x+1,y,1])
                neighbors.append([index1,index2])
            except:
                2+3
            try:
                index1 = lattice.index([x,y,0])
                index2 = lattice.index([x,y+1,1])
                neighbors.append([index1,index2])
            except:
                2+3
            try:
                index1 = lattice.index([x,y,0])
                index2 = lattice.index([x-1,y-1,1])
                neighbors.append([index1,index2])
            except:
                2+3
    return neighbors



print(build_lattice(1.5))
lat = build_lattice(1.5)
print(build_neighbor_table(lat))



