import numpy as np
from math import cos,sin
import math
import time

s0 = np.array([[1, 0],[ 0, 1]])
sx = np.array([[0, 1],[ 1, 0]])
sy = np.array([[0, -1j],[1j, 0]])
sz = np.array([[1, 0],[0, -1]])

q1 = np.array([0, -1.0])
q2 = np.array([math.sqrt(3)/2,1.0/2])
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
def coeffs(k):
    return np.linalg.solve([q1,q2],k)
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
            


