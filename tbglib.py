import numpy as np
from math import cos,sin
import math
import time
import matplotlib.pyplot as plt

s0 = np.array([[1, 0],[ 0, 1]])
sx = np.array([[0, 1],[ 1, 0]])
sy = np.array([[0, -1j],[1j, 0]])
sz = np.array([[1, 0],[0, -1]])

q1 = np.array([0, -1.0])
q2 = np.array([math.sqrt(3)/2,1.0/2])
q_matrix = np.transpose([q1,q2])
q3 = np.array([-math.sqrt(3)/2,1.0/2])
qs = [q1,q2,q3]
g1_coeff = [1,-1] #coords in terms of q1,q2
g1 = q1 -q2
g2_coeff = [1,2]
g2 = q1 + 2*q2
phi = 2 * np.pi /3
def norm(x):
    """ takes integer indices in teerms of q1, q2 returns norm"""
    return np.linalg.norm(x[0]*q1+x[1]*q2)
def coeffs(k,as_int=False):
    if as_int:
        return np.rint(np.linalg.solve(q_matrix,k)).astype(int)
    else:
        return np.linalg.solve(q_matrix,k)
def coords(coeffs):
    return q_matrix @ coeffs
def in_bz(k):
    """k in x,y basis"""
    if np.linalg.norm(k)>np.linalg.norm(k-g1):
        return False
    if np.linalg.norm(k)>np.linalg.norm(k+g1):
        return False
    if np.linalg.norm(k)>np.linalg.norm(k-g2):
        return False
    if np.linalg.norm(k)>np.linalg.norm(k+g2):
        return False
    if np.linalg.norm(k)>np.linalg.norm(k-g1-g2):
        return False
    if np.linalg.norm(k)>np.linalg.norm(k+g1+g2):
        return False
    return True

def build_bz():
    bz ={}
    
    bz["G_values"] = [g1,g2,-g1-g2,-g2,-g2,g1+g2] 
    # only closest in recip space, need to be symmetric for K' fudge to work
    bz["G_coeffs"] =[coeffs(k,as_int=True) for k in bz["G_values"]]
    bz["k_points"] = []

    N = 20
    for m in range(-N,N):
        for n in range(-N,N):
            if in_bz(m*g1+n*(g1+g2)+g1/2/N+(g1+g2)/2/N):
                    bz["k_points"].append(m*g1+n*(g1+g2)+g1/2/N+(g1+g2)/2/N)
    return bz

if __name__ =="__main__":
    print(g1,g2,g1+g2)
    print(coeffs(q1,True),coords([0,1]))
    print(build_bz())
    m = np.array(build_bz()["k_points"])
    print(m.shape)

    plt.scatter(m[:,0],m[:,1])
    plt.show()
    ks = np.zeros((20,20))
    for i in range(0,20):
        for j in range(0,20):
            if in_bz([i/5-2,j/5-2]):
                ks[i,j]=1
    print(ks)
    


