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
g_matrix = np.transpose([g1,g2])
print(g1,g2,g1+g2)

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
def in_bz_new(k):
    #basically checking if coeffs in (-0.5,0.5]x(-0.5,0.5]
    # for that need to first work with rationals
    coeffs = np.linalg.solve(g_matrix,k)
    for i in range(1,30):
        if np.isclose(i*coeffs,np.rint(i*coeffs)).all():
            break
    coeffs_int = np.rint(i*coeffs).astype(int)
    if 2*coeffs_int[0]>i or 2*coeffs_int[0]<-i+1:
        return False
    if 2*coeffs_int[1]>i or 2*coeffs_int[1]<-i+1:
        return False
    return True
def in_bz(k):
    """k in x,y basis"""
    if np.linalg.norm(k)>np.linalg.norm(k-g1):
        return False
    if np.linalg.norm(k)>np.linalg.norm(k+g1):
        return False
    if np.linalg.norm(k)>np.linalg.norm(k-g2):
        return False
    if np.linalg.norm(k)>=np.linalg.norm(k+g2):
        return False
    if np.linalg.norm(k)>=np.linalg.norm(k-g1-g2):
        return False
    if np.linalg.norm(k)>=np.linalg.norm(k+g1+g2):
        return False
    return True
def decompose(k):
    """ k = q + G, q in BZ, G in reciprocal lattice
        returns q,G """
    for i in range(-2,2):
        for j in range(-2,2):
            G = i*g1+j*g2
            if in_bz(k-G):
                return k-G,G
    print("Not in immediate neighborhood of bz:", k)

def dagger(A):
    return np.transpose(np.conjugate(A))
def closest_in_bz(k_points,point):
    min_val =  min(np.linalg.norm(np.array(k_points) - point ,axis=1))
    idx = (np.linalg.norm(np.array(k_points) - point ,axis=1)).argmin()
    for g in [np.array([0,0]),g1,g2,-g1-g2,-g1,-g2,g1+g2]:
        if min(np.linalg.norm(np.array(k_points) - point -g ,axis=1))<min_val:
            idx = (np.linalg.norm(np.array(k_points) - point -g ,axis=1)).argmin()
            print("hooray",idx)
    return idx
        

def build_bz(N=10):
    bz ={}
    
    bz["trajectory"]=[]
    bz["ticks_vals"]=["Ktop"]
    bz["ticks_coords"]=[0]
    bz["G_values"] = [np.array([0,0]),g1,g2,-g1-g2,-g1,-g2,g1+g2] 
    bz["G_neg_indices"] = [0,4,5,6,1,2,3] 
    # only closest in recip space, need to be symmetric for K' fudge to work
    bz["G_coeffs"] =[coeffs(k,as_int=True) for k in bz["G_values"]]
    bz["k_points"] = []
    bz["trajectory"] = [] #

    for m in range(-6*N-4,6*N+4):
        for n in range(-6*N-4,6*N+4):
            q = m/N*g1+n/N*(g1+g2)+g1/501/N+g2/501/N
            if in_bz(q):
                if m==n==0:
                    bz["index_0"]=len(bz["k_points"])
                bz["k_points"].append(q)
            #if in_bz(q+q1/N):
                #bz["k_points"].append(q+q1/N)

    idx = (np.linalg.norm(np.array(bz["k_points"]) - np.array([0,0]) ,axis=1)).argmin()
    bz["k_points_diff"] = np.array(bz["k_points"]) - bz["k_points"][idx]


    N_t = 20
    for i in range(N_t+1):
        point = -q1 + 2*q1*(i/N_t)
        idx = closest_in_bz(bz["k_points"],point)
        bz["trajectory"].append(idx)
    bz["ticks_vals"].append("K")
    bz["ticks_coords"].append(int(i/2))
    bz["ticks_vals"].append("-Ktop=Gamma")
    bz["ticks_coords"].append(len(bz["trajectory"]))

    for i in range(1,N_t+1):
        point = q1 -(2*q1+q2)*i/N_t
        idx = closest_in_bz(bz["k_points"],point)
        bz["trajectory"].append(idx)
    bz["ticks_vals"].append("Gamma")
    bz["ticks_coords"].append(len(bz["trajectory"]))
    for i in range(1,N_t+1):
        point = q1 -(2*q1+q2)*i/N_t
        idx = closest_in_bz(bz["k_points"],point)
        bz["trajectory"].append(idx)
    bz["ticks_vals"].append("Gamma")
    bz["ticks_coords"].append(len(bz["trajectory"]))
    for i in range(1,N_t+1):
        point = -q1 -q2 +q2*i/N_t
        idx = closest_in_bz(bz["k_points"],point)
        bz["trajectory"].append(idx)
    bz["ticks_vals"].append("Ktop")
    bz["ticks_coords"].append(len(bz["trajectory"]))
    return bz

if __name__ =="__main__":
    #print(g1,g2,g1+g2)
    #print(coeffs(q1,True),coords([0,1]))
    print(coeffs([1.732,0]))
    #print(build_bz())
    m = np.array(build_bz(10)["k_points"])
    print(m.shape)

    plt.scatter(m[:,0],m[:,1])
    plt.show()
    ks = np.zeros((20,20))
    for i in range(0,20):
        for j in range(0,20):
            if in_bz([i/5-2,j/5-2]):
                ks[i,j]=1
    print(ks)
