import numpy as np
from math import cos,sin
import math
import time
import matplotlib.pyplot as plt
from functools import reduce

s0 = np.array([[1, 0],[ 0, 1]])
sx = np.array([[0, 1],[ 1, 0]])
sy = np.array([[0, -1j],[1j, 0]])
sz = np.array([[1, 0],[0, -1]])
theta = 2*math.pi/3
c3_rot = np.array([[cos(theta),-sin(theta)],[sin(theta),cos(theta)]])

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
g_s =[g1,g2,-g1-g2,-g1,-g2,g1+g2] 

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
def valley_inv(P):
    m = np.diag([0+1j,0+1j,0-1j,0-1j,0+1j,0+1j,0-1j,0-1j])
    s = reduce(lambda x, k: x +
            np.linalg.norm(m\
            @ P[k] @ np.conjugate(m) \
            - P[k]),range(len(P)))
    print("\nHF solution valley invariance:",s)
def in_bz(k,size_bz=None):
    size_bz = None
    if np.linalg.norm(k-q1)<1e-7:
        return True
    if np.linalg.norm(k+q1)<1e-7:
        return True
    #print("\n k:", k)
    """k in x,y basis"""
    for g in g_s[:3]:
        d = np.dot(k,g)
        if d<0:
            continue
        if size_bz==None:
            for i in range(1,50):
                if np.isclose(i*d,np.rint(i*d)):
                    break
        else:
            i = size_bz
        coeffs_int = np.rint(i*d).astype(int)
        #print(coeffs_int,i)
        if 2*coeffs_int > 3*i:
            return False
    for g in g_s[3:]:
        d = np.dot(k,g)#/np.linalg.norm(g)
        #print(d)
        if d<0:
            continue
        if size_bz==None:
            for i in range(1,50):
                #print(i)
                if np.isclose(i*d,np.rint(i*d)):
                    break
        else:
            i = size_bz
        coeffs_int = np.rint(i*d).astype(int)
        if 2*coeffs_int >= 3*i:
            return False
    return True 
def in_bz_old(k):
    if np.linalg.norm(k)>=np.linalg.norm(k-g1):
        return False
    if np.linalg.norm(k)>=np.linalg.norm(k+g1):
        return False
    if np.linalg.norm(k)>=np.linalg.norm(k-g2):
        return False
    if np.linalg.norm(k)>=np.linalg.norm(k+g2):
        return False
    if np.linalg.norm(k)>=np.linalg.norm(k-g1-g2):
        return False
    if np.linalg.norm(k)>=np.linalg.norm(k+g1+g2):
        return False
    return True
def decompose(k,N=None):
    """ k = q + G, q in BZ, G in reciprocal lattice
        returns q,G """
    for i in range(-4,4):
        for j in range(-4,4):
            G = i*g1+j*g2
            if in_bz(k-G,N):
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
    return idx
def is_equilateral_triangle(a,b,c):
    if abs(np.linalg.norm(a-b)-np.linalg.norm(a-c))>1e-5:
        return False
    if abs(np.linalg.norm(c-b)-np.linalg.norm(a-c))>1e-5:
        return False
    if abs(np.linalg.norm(a-b)-np.linalg.norm(b-c))>1e-5:
        return False
    return True
def eval(k_eval, data, k_points):
    """ evaluates data at k_eval by interpolating"""
    #print(k_eval)
    k_eval,G = decompose(k_eval)
    k_eval = k_eval  
    idx = np.argpartition(np.linalg.norm(np.array(k_points) - k_eval - np.array([0.000000432,0.0000000343]) ,axis=1),3)
    a = k_points[idx[0]]
    b = k_points[idx[1]]
    c = k_points[idx[2]]
    f0 = np.array(data)[idx[0]]
    f1 = np.array(data)[idx[1]]
    f2 = np.array(data)[idx[2]]
    if abs(f0[0]-f1[0])>30:
        if False:
            print("Warning, values at closely lying k points very differnt",f0,f1)
            print("k_eval is:", k_eval, " and a -b:", np.linalg.norm(a-b))
            print("a is:",a,"and b is:", b)
    center =1/3*( a + b + c )
    l = np.linalg.norm(a-center)
    if is_equilateral_triangle(a,b,c):
        res =  2/3*f0*np.dot(a - center,k_eval-center)/l**2+\
         2/3*f1*np.dot(b - center,k_eval-center)/l**2+\
         2/3*f2*np.dot(c - center,k_eval-center)/l**2+\
         1/3*(f1+f2+f0)
    else:
        G_vals = [np.array([0,0]),g1,g2,-g1-g2,-g1,-g2,g1+g2,g1-g2,g2-g1] 
        data_new = np.concatenate(tuple([data for m in range(len(G_vals))]))
        k_points_new = np.concatenate(tuple([np.array(k_points)+G_vals[m] for m in range(len(G_vals))]))
        return eval(k_eval, data_new,k_points_new)
    return res
def build_bz(N=10, shifted = False):
    """shifted has c3 symmetry, G's also need to respect c3  """
    print("Building bz")
    if shifted and N%3!=0:
        print("Warning, one of the points lies on q0 or -q0")
        print("It is best to avoid those since then one avoids degeneracies")
    bz ={}
    
    bz["trajectory"]=[]
    bz["ticks_vals"]=["-Ktop = Gamma"]
    bz["ticks_coords"]=[0]
    bz["G_values"] = [np.array([0,0]),g1,g2,-g1-g2,-g1,-g2,g1+g2] 
    bz["G_neg_indices"] =[0,4,5,6,1,2,3] 
    if False:
        nbz = 3 
        bz["G_values"] =[np.array([0,0])] 
        bz["G_neg_indices"] =[0] 
        for m in range(-nbz,nbz+1):
            for n in range(nbz+1):
                if m>-1 and n==0:
                    continue
                q = m*g1+n*g2
                if np.linalg.norm(q)>nbz*math.sqrt(3):
                    continue
                bz["G_values"].append(q)
                bz["G_values"].append(-q)
                bz["G_neg_indices"].append(len(bz["G_values"])-1)# [0,4,5,6,1,2,3,8,7] 
                bz["G_neg_indices"].append(len(bz["G_values"])-2)# [0,4,5,6,1,2,3,8,7] 


    # only closest in recip space, need G_values to be inversion symmetric for K' fudge to work
    bz["G_coeffs"] =[coeffs(k,as_int=True) for k in bz["G_values"]]
    bz["k_points"] = []
    bz["trajectory"] = [] #
    bz["trajectory_points"] = []
    bz["c3_indices"] = []
    bz["c3_indices_of_Gs"] = []

    print("A")
    for m in range(-6*N-4,6*N+4):
        for n in range(-6*N-4,6*N+4):
            if shifted:
                q = m/N*g1+n/N*(g1+g2)+ q1/N#g1/501/N+g2/501/N
            else:
                q = m/N*g1+n/N*(g1+g2)# + q1/N#g1/501/N+g2/501/N
            if in_bz(q,N):
                if m==n==0:
                    bz["index_0"]=len(bz["k_points"])
                bz["k_points"].append(q)

    idx = (np.linalg.norm(np.array(bz["k_points"]) - np.array([0,0]) ,axis=1)).argmin()
    bz["k_points_diff"] = np.array(bz["k_points"]) - bz["k_points"][idx]
    print("B")

    for k in range(len(bz["k_points"])):
        #workis
        k_rot, G = decompose(np.dot(c3_rot, bz["k_points"][k]),N) #k_rot + G= c3 @ k
        idx_G = (np.linalg.norm(np.array(bz["G_values"]) - G ,axis=1)).argmin()
        if idx_G!=0:
            continue
            #print("WHAAAT",idx_G,k_rot)
            #print("G is:" , G, "k_rot is:", k_rot, "G+k_rot", G+k_rot)

        idx = (np.linalg.norm(np.array(bz["k_points"]) - k_rot ,axis=1)).argmin()
        if np.linalg.norm(bz["k_points"][idx]-k_rot)>1e-7:
            print("Warning, k rot",k_rot, "thought closest is",bz["k_points"][idx])

        #print(bz["k_points"][idx],k_rot)
        #print(bz["k_points"][idx],k_rot)
        bz["c3_indices"].append(idx)
        bz["c3_indices_of_Gs"].append(idx_G)

    N_t = 40
    for i in range(N_t+1):
        point = -3*q1 + 2*q1*(i/N_t)# + np.array([0.000001237,0.000001143])
        idx = closest_in_bz(bz["k_points"],point)
        bz["trajectory"].append(idx)
        bz["trajectory_points"].append(point)
    bz["ticks_vals"].append("Gamma")
    bz["ticks_coords"].append(int(i/2))
    bz["ticks_vals"].append("K_top")
    bz["ticks_coords"].append(len(bz["trajectory"])-1)

    for i in range(1,N_t+1):
        point = -q1 +  2* q1*(i/N_t) #+ np.array([0.000001237,0.000001143])
        idx = closest_in_bz(bz["k_points"],point)
        bz["trajectory"].append(idx)
        bz["trajectory_points"].append(point)
    bz["ticks_vals"].append("-Ktop=Gamma")
    bz["ticks_coords"].append(len(bz["trajectory"])-1)


    for i in range(1,N_t+1):
        point = q1 -(2*q1+q2)*i/N_t #+ np.array([0.00000032134,0.000001143])
        idx = closest_in_bz(bz["k_points"],point)
        bz["trajectory"].append(idx)
        bz["trajectory_points"].append(point)
    bz["ticks_vals"].append("Gamma")
    bz["ticks_coords"].append(len(bz["trajectory"])-1)
    for i in range(1,N_t+1):
        point = q1 -(2*q1+q2)*i/N_t #+ np.array([0.0000012137,0.000001143])
        idx = closest_in_bz(bz["k_points"],point)
        bz["trajectory"].append(idx)
        bz["trajectory_points"].append(point)
    bz["ticks_vals"].append("Gamma")
    bz["ticks_coords"].append(len(bz["trajectory"])-1)
    for i in range(1,N_t+1):
        point = -q1 -q2 +q2*i/N_t #+ np.array([0.0000012139,0.00000145])
        idx = closest_in_bz(bz["k_points"],point)
        bz["trajectory"].append(idx)
        bz["trajectory_points"].append(point)
    bz["ticks_vals"].append("Ktop")
    bz["ticks_coords"].append(len(bz["trajectory"])-1)
    return bz

if __name__ =="__main__":
    print(g1,g2,g1+g2)
    print(coeffs(q1,True),coords([0,1]))
    print(coeffs([1.732,0]))
    #print(build_bz())
    m = np.array(build_bz(3,True)["k_points"])
    print("Shape of M", m.shape)

    plt.scatter(m[:,0],m[:,1])
    #plt.scatter(m[:,0],-m[:,1]-q1[1],marker = "x")
    bz = build_bz(3,True)
    
    print("Lenght", len(bz["G_values"][1:]))
    for g in bz["G_values"][1:3]:
        plt.scatter((m+g)[:,0],(m+g)[:,1],marker="x")
    plt.grid()
    plt.show()

    ks = np.zeros((20,20))
    for i in range(0,20):
        for j in range(0,20):
            if in_bz([i/5-2,j/5-2]):
                ks[i,j]=1
    print(ks)
