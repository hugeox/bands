import numpy as np
from math import cos,sin
import math
import time
import matplotlib.pyplot as plt
import generate_band_structure as gbs
import tbglib
import h5py
import json
from functools import reduce

""" solve a self-consistent Hartree-Fock at given filling and save into file"""

def V_coulomb(q_vec):
    #returns V(q) in units of meV nm^2
    q=np.linalg.norm(q_vec)
    d_s = 40 #screening lenght in nm
    scaling_factor = 2* sin(1.09*np.pi/180)*\
                4*np.pi/(3*math.sqrt(3)*0.246)
    epsilon=1/0.06
    if q*scaling_factor*d_s<0.01:
        return 1439.96*d_s*4*np.pi/epsilon #in eV nm^2
    else:
        return 1439.96*2*np.pi*math.tanh(q*scaling_factor*d_s)/\
                (q*scaling_factor*epsilon)

def read_hf_from_file(filename):
    hf_solution={}
    model_params = {"theta" : 1.09 * np.pi / 180, #twist angle in radians
                    "w_AA" :80, #in meV
                    "w_AB" : 110,#110 #in meV
                    "v_dirac" : int(19746/2), #v_0 k_D in meV
                    "epsilon" : 1/0.06,
                    "d_s": 40, #screening length in nm
                    "scaling_factor": 2* sin(1.09*np.pi/180)*\
                    4*np.pi/(3*math.sqrt(3)*0.246) ,
                    "single_gate_screening": False, #single or dual gate screening?
                    "q_lattice_radius": 10,
                    "V_coulomb" : V_coulomb #V_q
                    }
    f_in = h5py.File(filename, 'r')
    for k in f_in.keys():
        print(k)
        hf_solution[k] = f_in[k][...]
    SIZE_BZ = f_in.attrs["size_bz"]
    bz = tbglib.build_bz(SIZE_BZ)
    hf_solution["bz"] = bz 
    for key in f_in.attrs.keys():
        if key !="V_coulomb":
            model_params[key] = f_in.attrs[key] 
    f_in.close()
    hf_solution["model_params"] = model_params
    hf_solution["k_points"] = bz["k_points"]
    hf_solution["N"] = len(bz["k_points"])
    return hf_solution

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

def C2T_eigenvalue(state):
    """ actually makes the eigenvalue real"""
    a = 0
    eigvalue = np.conjugate(state[a+1])/state[a]
    #eigenvalue = s/counter
    eigvalue2 = np.conjugate(state[a+3])/state[a+2]
    if abs(abs(eigvalue)-1)>0.000001:
        print("C2T Eigvalue is, and its norm is not one: ", eigvalue)
    if abs(eigvalue-eigvalue2)>0.000001:
        print("C2T Eigvalue is, and is not equal with second method : ", eigvalue)
    #np.multiply(np.sqrt(eigvalue),state,state)
    #eigvalue = np.conjugate(state[a+1])/state[a]
    #print("Should be 1:", eigvalue)
    return eigvalue
def c3_eval(state, rotated_state,valley = False):
    a = 0
    #rot = cos(tbglib.theta)*tbglib.s0 - (0+1j)*sin(tbglib.theta)*tbglib.sz
    new_state = [0,0]
    new_state[1] = np.exp((0+1j)*2*tbglib.theta)*state[1]
    new_state[0] = np.exp((0+1j)*1*tbglib.theta)*state[0]
    if valley:
        new_state[1] = np.exp((0+1j)*1*tbglib.theta)*state[1]
        new_state[0] = np.exp((0+1j)*2*tbglib.theta)*state[0]
    eigvalue = new_state[0]/rotated_state[a]
    eigvalue2 = new_state[1]/rotated_state[a+1]
    #print("new state", new_state)
    #print("rotated state", rotated_state[:2])
    if abs(abs(eigvalue)-1)>0.000001:
        print("c3 Eigvalue is, and its norm is not one: ", eigvalue)
    if abs(eigvalue-eigvalue2)>0.000001:
        print("c3 Eigvalue is, and is not equal with second method : ", eigvalue,eigvalue2)
    return eigvalue
def all_c2t_evals(states):
    arr = [C2T_eigenvalue(states[0]),C2T_eigenvalue(states[1])]
    return arr
    
def overlap(state1,state2,shift_matrix):
    """dot product <state1| e^iGr|state2>"""
    return np.vdot(state1,np.dot(shift_matrix,state2))
def build_overlaps(bz,model_params):
    """ singleparticle energies and overlaps"""
    G_coeffs = bz["G_coeffs"]
    k_points = bz["k_points"]
    q_lattice_radius = model_params["q_lattice_radius"]
    N_f = model_params["N_f"] #N_flavors
    lattice, neighbor_table = gbs.build_lattice_and_neighbor_table(q_lattice_radius)
    es = []
    ss = []
    s_matrices = []
    c2t_evals = []
    c3_evals = []
    projected_szs = []
    sz = np.kron(np.identity(len(lattice)),tbglib.sz)
    if model_params["valley"]:
        ss_prime = []
        s_matrices_prime = []
    for k in k_points:
        print("evaluating energies for k: ", k)
        energies, states = gbs.find_energies(k,
            params = model_params, N_bands = 2, 
            lattice = lattice,
            neighbor_table = neighbor_table,return_states = True )
        es.append([energies[0],energies[1]])
        c2t_evals.append(all_c2t_evals(states))
        ss.append(states)
        if model_params["valley"]:
            energies_prime, states_prime = gbs.find_energies(-k,
                params = model_params, N_bands = 2, 
                lattice = lattice,
                neighbor_table = neighbor_table,return_states = True )
            #allways remember coefficient at G at this valley is u_-G
            ss_prime.append(np.conjugate(states_prime))
            es[-1].extend(energies_prime)
            c2t_evals[-1].extend(all_c2t_evals(ss_prime[-1]))
        #print("SP ENERGIES at valley", energies)
    for G in G_coeffs:
        s_matrices.append(shift_matrix(-G,lattice)) #TODO: Check sign of G
        if model_params["valley"]:
            s_matrices_prime.append(shift_matrix(G,lattice)) #K' fudge
    for k in range(len(k_points)):
        #print(k_points[k])
        idx_rotated = bz["c3_indices"][k]
        idx_of_G = bz["c3_indices_of_Gs"][k]
        #print("IDX OF G", idx_of_G)
        c3_evals.append([c3_eval(ss[k][0],np.dot(s_matrices[idx_of_G],  ss[idx_rotated][0])),
                c3_eval(ss[k][1],np.dot(s_matrices[idx_of_G], ss[idx_rotated][1]))])
        if model_params["valley"]:
            #print("Second valley")
            #print(es[k])
            c3_evals[-1].append(c3_eval(ss_prime[k][0],np.dot(s_matrices_prime[idx_of_G],  ss_prime[idx_rotated][0]),True))
            c3_evals[-1].append(c3_eval(ss_prime[k][1],np.dot(s_matrices_prime[idx_of_G],  ss_prime[idx_rotated][1]),True))
    
    overlaps = np.zeros((len(G_coeffs),len(k_points),
                        len(k_points),N_f,N_f),dtype = complex)
    projected_szs = np.zeros((len(k_points),N_f,N_f),dtype = complex)
    for i in range(len(k_points)):
        print(i)
        for j in range(len(k_points)):
            for g in range(len(G_coeffs)):
                for m in range(2):
                    for n in range(2):
                        overlaps[g,i,j,m,n] = overlap(ss[i][m],ss[j][n],s_matrices[g])
                        if model_params["valley"]:
                            overlaps[g,i,j,m+2,n+2] = overlap(ss_prime[i][m],ss_prime[j][n],s_matrices_prime[g])
    for i in range(len(k_points)):
        for m in range(2):
            for n in range(2):
                if m==n:
                    projected_szs[i,m,n]= np.vdot(ss[i][m],np.dot(sz,ss[i][n]))
                    if model_params["valley"]:
                        projected_szs[i,m+2,n+2]=np.vdot(ss_prime[i][m],np.dot(sz,ss_prime[i][n]))
                else: # botch to normalize
                    projected_szs[i,m,n]=np.exp(1j*np.angle( np.vdot(ss[i][m],np.dot(sz,ss[i][n]))))
                    if model_params["valley"]:
                        projected_szs[i,m+2,n+2]=np.exp(1j*np.angle(np.vdot(ss_prime[i][m],np.dot(sz,ss_prime[i][n]))))
    print("Projected szs:", projected_szs[0])
    if model_params["spin"]:
        #double es, c2t_evals, c3_evals for second spin species
        es = np.concatenate((es,es),axis=1)
        c2t_evals = np.concatenate((c2t_evals,c2t_evals),axis=1)
        c3_evals = np.concatenate((c3_evals,c3_evals),axis=1)
        if model_params["valley"]:
            overlaps[:,:,:,4:,4:]=overlaps[:,:,:,:4,:4]
            projected_szs[:,4:,4:]=projected_szs[:,:4,:4]
        else:
            overlaps[:,:,:,2:,2:]=overlaps[:,:,:,:2,:2]
            projected_szs[:,2:,2:]=projected_szs[:,:2,:2]
        #print(overlaps[0,1,0,:,:])
        #print(overlaps[0,0,0,:,:])
    return es, overlaps, c2t_evals, c3_evals,projected_szs
def v_hf(bz,overlaps,model_params,P,V_c):
    G_coeffs = bz["G_coeffs"]
    G_s = bz["G_values"]
    k_points = bz["k_points"]
    V_coulomb = V_c# model_params["V_coulomb"]
    N_f = model_params["N_f"] #N_flavors
    V_hf = np.zeros((len(k_points),N_f,N_f),dtype = complex)

    direct_potential = [] #at G
    for g in range(len(G_coeffs)):
        temp = 0
        for k in range(len(k_points)):
            temp = temp + \
                 np.trace(P[k] @ np.conjugate(overlaps[g,k,k,:,:]))
        if g==0 and True: #remove 0 component of V
            temp = 0
        direct_potential.append(temp)
    scaling_factor =  2* sin(model_params["theta"])*\
                    4*np.pi/(3*math.sqrt(3)*0.246)
    for k in range(len(k_points)):
        temp = np.zeros((N_f,N_f),dtype = complex)
        for g in range(len(G_coeffs)):
            temp = temp + \
                    direct_potential[g]*V_coulomb(G_s[g])*\
                    overlaps[g,k,k,:,:]
            for l in range(len(k_points)):
                temp = temp - \
                    V_coulomb(G_s[g]+k_points[l]-k_points[k])*\
                    overlaps[g,k,l,:,:]@\
                            np.matrix.transpose(P[l])@\
                            tbglib.dagger(
                           overlaps[g,k,l,:,:])



        V_hf[k,:,:]=scaling_factor**2*temp/(len(k_points)*1.5*math.sqrt(3))
                        #area of hexagon
    return V_hf

def hf_energy_total(bz,energies, overlaps, model_params, P,V_coulomb):
    temp = 0
    v_mf = v_hf(bz,overlaps,model_params, P,V_coulomb)
    en = []
    for e in energies:
        en.append(np.diag(e))
    en = np.array(en) 
    k_points = bz["k_points"]
    for k in range(len(k_points)):
        temp = temp + np.trace(np.transpose(P[k]) @ (en[k] + 0.5*v_mf[k]))
    return temp
    
def iterate_hf(bz,energies, overlaps, model_params, P,V_c, k_dep_filling = False):
    en = []
    for e in energies:
        en.append(np.diag(e))
    P_0 = np.zeros(np.array(P).shape)
    for k in range(len(energies)):
        for i in range(0,len(P[0]), 2):
            P_0[k,i,i]=1
    h_mf = np.array(en) + v_hf(bz,overlaps,model_params, P,V_c)
                #v_hf(bz,overlaps,model_params, P_0,V_c) #subtract uniform cn solution
    G_coeffs = bz["G_coeffs"]
    G_s = bz["G_values"]
    k_points = bz["k_points"]
    flavors = len(P[0])
    hf_energies = []
    hf_states = []
    total_fill = 0
    k_fillings = np.zeros(len(k_points),dtype = int)
    P_new = np.zeros(np.array(P).shape,dtype = complex)
    for k in range(len(k_points)):
        filling = int(round(np.real(np.trace(P[k]))))
        energies, states = np.linalg.eigh(h_mf[k,:,:])
        m = np.zeros(flavors,dtype = complex)
        m[:filling]=1
        total_fill = total_fill + filling
        if k_dep_filling == False:
            P_new[k,:,:] = np.conjugate(states) @\
                         np.diag(m) @  np.transpose(states)
        hf_energies.append(energies)
        hf_states.append(states)
    if k_dep_filling:
        print(total_fill)
        arr = np.transpose(hf_energies)
        idx = np.argpartition(arr.ravel(), total_fill)
        for i in range(total_fill):
            id_k = idx[i]%len(k_points)
            k_fillings[id_k] = k_fillings[id_k] + 1
        for k in range(len(k_points)):
            filling = k_fillings[k]
            m = np.zeros(flavors,dtype = complex)
            m[:filling]=1
            P_new[k,:,:] = np.conjugate(hf_states[k]) @\
                         np.diag(m) @  np.transpose(hf_states[k])
        print(k_fillings)

    return P_new, hf_energies, hf_states

class hf_solver(object):
    def __init__(self,model_params,P_0 = None):
        """ build using params, takes time! """
        if type(model_params)==str:
            return self.load(model_params)
        self.params = model_params
        self.bz = tbglib.build_bz(self.params["size_bz"],self.params["shifted_bz"])
        N_dof = 2
        if model_params["spin"]:
            N_dof = 2*N_dof
        if model_params["valley"]:
            N_dof = 2*N_dof
        self.params["N_f"] = N_dof
        print("N_flavors", N_dof)
        a = np.zeros((N_dof))
        a[0] = 1
        P_k=np.diag(a)
        print(P_k)
        self.N_k = len(self.bz["k_points"])
        self.P_0 = [P_k.copy() for k in range(self.N_k)]#default P_0
        self.P = [P_k.copy() for k in range(self.N_k)]#default P_0
        print("Number of points is:", self.N_k)
        self.sp_energies, self.overlaps,\
            self.c2t_eigenvalues, self.c3_eigenvalues, self.projected_szs= \
                build_overlaps(self.bz,model_params)
    def load(self,filepath):
        """ load from hdf5 file """
        hf_solution={}
        model_params = {}
        f_in = h5py.File(filepath, 'r')
        for k in f_in.keys():
            #print(k)
            hf_solution[k] = f_in[k][...]
        
        SIZE_BZ = f_in.attrs["size_bz"]
        print("Size of brillouin-zone: ", SIZE_BZ)
        for key in f_in.attrs.keys():
            model_params[key] = f_in.attrs[key] 
            if key == "description":
                print("Description:", f_in.attrs[key])
        bz = tbglib.build_bz(SIZE_BZ,model_params["shifted_bz"])
        self.bz = bz 
        f_in.close()
        self.params = model_params
        N_dof = 2
        try:
            if model_params["spin"]:
                N_dof = 2*N_dof
            if model_params["valley"]:
                N_dof = 2*N_dof
        except:
            2+3
        try:
            self.projected_szs = hf_solution["projected_szs"]
        except:
            2+3
        self.params["N_f"] = N_dof
        self.overlaps = hf_solution["overlaps"]
        self.sp_energies = hf_solution["sp_energies"]
        self.hf_eigenvalues = hf_solution["hf_eigenvalues"]
        self.hf_eigenstates = hf_solution["hf_eigenstates"]
        self.P_0 = hf_solution["P_0"]
        self.P = hf_solution["P_hf"]
        self.c2t_eigenvalues = hf_solution["c2t_eigenvalues"]
        self.c3_eigenvalues = hf_solution["c3_eigenvalues"]
        print("first c3 eigenvalue",self.c3_eigenvalues[0])
        self.N_k = len(bz["k_points"])
    def reset_P(self,P_in = None):
        if type(P_in) is None:
            self.P = self.P_0.copy()
        else:
            self.P_0 = P_in
            self.P = P_in
    def set_state(self,break_c2t = False, break_c3 = False, coherent = False):
        if coherent:
            self.params["description"] ="Coherent state"
            return self.set_coherent_state()
        else:
            filling = self.params["filling"]
            N_f = self.params["N_f"]
            N_k = self.N_k
            self.params["description"] ="break_c3: {}, break_c2t:{}".format(break_c3,break_c2t)
            self.reset_P(tbglib.create_state(N_k,N_f,filling,break_c2t,break_c3))
    def set_coherent_state(self,angle = 0):
        N_f = self.params["N_f"]
        A =  np.zeros((N_f,N_f),dtype=complex)
        P_1=[]
        for k in range(self.N_k):
            evals, states = np.linalg.eigh(self.projected_szs[k,:,:])
            P_1.append(0.5*np.identity(N_f)+0.5*np.conjugate(states)@ \
                    np.kron(np.identity(int(N_f/2)),tbglib.sy)@\
            #        np.kron(tbglib.sz,cos(angle)*tbglib.sx+ sin(angle)*tbglib.sy)@\
                    np.transpose(states))
            # sets QH state
            #P_1.append(0.5*np.identity(N_f)+0.5*np.conjugate(states)@ \
            #        np.diag([1,1,-1,-1])@\
            #        np.transpose(states))
        self.reset_P(P_1)

    def V_coulomb(self,q_vec):
        #returns V(q) in units of meV nm^2
        q=np.linalg.norm(q_vec)
        d_s = self.params["d_s"] #screening lenght in nm
        scaling_factor = 2* sin(self.params["theta"])*\
                    4*np.pi/(3*math.sqrt(3)*0.246)
        epsilon=self.params["epsilon"]
        if q*scaling_factor*d_s<0.01:
            return 1439.96*d_s*4*np.pi/epsilon #in eV nm^2
        else:
            return 1439.96*2*np.pi*math.tanh(q*scaling_factor*d_s)/\
                    (q*scaling_factor*epsilon)
    def iterate_hf(self,check_c2t = False,check_c3 = False,impose_c2t = False,
                        impose_c3 = False):
        if check_c3:
            s = reduce(lambda x, k: x + np.linalg.norm(np.diag(self.c3_eigenvalues[k])\
                            @ self.P[self.bz["c3_indices"][k]] @ \
                            np.diag(np.conjugate(self.c3_eigenvalues[k])) \
                            - self.P[k]),range(self.N_k))
            print("\nHF solution c3 invariance:",s)
        P, energies,states = iterate_hf(self.bz,self.sp_energies,
                self.overlaps,
                self.params, self.P,self.V_coulomb,False)
        dist = np.linalg.norm(np.array(P).ravel()-np.array(self.P).ravel())
        print("HF distance",dist)
        if impose_c2t:
            #print(P[0])
            for k in range(self.N_k):
                P[k] = 0.5*np.diag(np.conjugate(self.c2t_eigenvalues[k])) \
                        @ np.transpose(P[k]) @ np.diag(self.c2t_eigenvalues[k])+ 0.5 * P[k]
            #print(P[0]@P[0],P[0])
        if check_c2t:
            s = reduce(lambda x, k: x +
                    np.linalg.norm(np.diag(self.c2t_eigenvalues[k])\
                    @ P[k] @ np.diag(np.conjugate(self.c2t_eigenvalues[k])) \
                    - np.transpose(P[k])),range(self.N_k))
            print("\nHF solution c2t invariance:",s)
        if check_c3:
            s = reduce(lambda x, k: x + np.linalg.norm(np.diag(self.c3_eigenvalues[k])\
                            @ P[self.bz["c3_indices"][k]] @ \
                            np.diag(np.conjugate(self.c3_eigenvalues[k])) \
                            - P[k]),range(self.N_k))
            print("\nHF solution c3 invariance:",s)
        if impose_c3:
            for k in range(self.N_k):
                P[self.bz["c3_indices"][k]] = np.diag(self.c3_eigenvalues[self.bz["c3_indices"][k]])\
                            @ P[self.bz["c3_indices"][self.bz["c3_indices"][k]]] @ \
                            np.diag(np.conjugate(self.c3_eigenvalues[self.bz["c3_indices"][k]]))
                P[k] = np.diag(self.c3_eigenvalues[k])\
                            @ P[self.bz["c3_indices"][k]] @ \
                            np.diag(np.conjugate(self.c3_eigenvalues[k]))
        self.P = P
        self.hf_eigenvalues = energies
        self.hf_eigenstates = states
        return dist
    def eval(self,k):
        return tbglib.eval(k,self.hf_eigenvalues,self.bz["k_points"])
    def eval_sp(self,k):
        return tbglib.eval(k,self.sp_energies,self.bz["k_points"])
    def check_v_c2t_invariance(self):
        k = 0
        v = v_hf(self.bz,self.overlaps,self.params,self.P,self.V_coulomb)
        print("V(k) c2t invar: ",np.linalg.norm(np.diag(self.c2t_eigenvalues[k]) @ np.conjugate(v[k])@ np.diag(np.conjugate(self.c2t_eigenvalues[k])) - v[k]))
        v = v_hf(self.bz,self.overlaps,self.params,self.P_0,self.V_coulomb)
        print("Initial V(k) c2t invar: ",np.linalg.norm(np.diag(self.c2t_eigenvalues[k]) @ np.conjugate(v[k])@ np.diag(np.conjugate(self.c2t_eigenvalues[k])) - v[k]))
        print("Initial V(k) c3 invar: ",
                        np.linalg.norm(np.diag(self.c3_eigenvalues[k]) @ \
                        np.array( v[k])@ \
                        np.diag(np.conjugate(self.c3_eigenvalues[k])) -v[self.bz["c3_indices"][k]]))
        v = v_hf(self.bz,self.overlaps,self.params,self.P,self.V_coulomb)
        print("V(k) c3 invar: ",
                        np.linalg.norm(np.diag(self.c3_eigenvalues[k]) @ \
                        np.array(v[self.bz["c3_indices"][k]])@ \
                        np.diag(np.conjugate(self.c3_eigenvalues[k])) - v[k]))
    def save(self,filepath):
        f_out = h5py.File(filepath, 'w')
        f_out.create_dataset("overlaps", data = self.overlaps)
        f_out.create_dataset("sp_energies", data = self.sp_energies)
        f_out.create_dataset("P_hf", data = self.P)
        f_out.create_dataset("P_0", data = self.P_0)#save also inital guess
        f_out.create_dataset("hf_eigenvalues", data = self.hf_eigenvalues)
        f_out.create_dataset("hf_eigenstates", data = self.hf_eigenstates)
        f_out.create_dataset("c2t_eigenvalues", data = self.c2t_eigenvalues)
        f_out.create_dataset("c3_eigenvalues", data = self.c3_eigenvalues)
        try:
            f_out.create_dataset("projected_szs", data = self.projected_szs)
        except:
            1+1
        for key in self.params.keys():
            if key !="V_coulomb":
                f_out.attrs[key] = self.params[key]
        f_out.close()
    def plot3d(self):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        X = np.arange(-1.5000023, 2, 0.15)
        Y = np.arange(-1.500000054, 2, 0.15)
        Z = np.array([[self.eval(np.array([x,y]))[0] for x in X] for y in Y])
        X, Y = np.meshgrid(X, Y)
        surf = ax.plot_surface(X,
                Y,Z)
        X = np.arange(-1.5000023, 2, 0.15)
        Y = np.arange(-1.50000054, 2, 0.15)
        Z = np.array([[self.eval(np.array([x,y]))[1] for x in X] for y in Y])
        X, Y = np.meshgrid(X)
        surf = ax.plot_surface(X,
                Y,Z)
    def hf_energy(self):
        return hf_energy_total(self.bz,self.sp_energies, self.overlaps,
self.params, self.P,self.V_coulomb)

if __name__ == "__main__":
    #execution

    model_params = {"theta" : 1.09 * np.pi / 180, #twist angle in radians
                    "w_AA" :80, #in meV
                    "w_AB" : 110,#110 #in meV
                    "v_dirac" : int(19746/2), #v_0 k_D in meV
                    "epsilon" : 1/0.06,
                    "d_s": 40, #screening length in nm
                    "scaling_factor": 2* sin(1.05*np.pi/180)*\
                    4*np.pi/(3*math.sqrt(3)*0.246) , #this will actually be computed from theta, 0.246nm = lattice const. of graphene
                    "single_gate_screening": False, #single or dual gate screening?
                    "q_lattice_radius": 12,
                    "size_bz": 12,
                    "shifted_bz": True,
                    "description": " only valley,smaller angle",
                    "V_coulomb" : V_coulomb,
                    "filling": -3,
                    "hf_iters": 20,
                    "spin": False,
                    "valley": True
                    }

    solver = hf_solver(model_params,None)

    id =7
    print(id)

    for m in range(solver.params["hf_iters"]):
        solver.iterate_hf(True,True,False,False)
    solver.save("data/hf_{}.hdf5".format(id))
    adfds
    for k in range(solver.N_k-1):
        print("\n")
        print("one",solver.hf_eigenvalues[k])
        print("rotated", solver.hf_eigenvalues[solver.bz["c3_indices"][k]])
        g1 = solver.bz["G_values"][solver.bz["c3_indices_of_Gs"][k]]
        g2 = solver.bz["G_values"][solver.bz["c3_indices_of_Gs"][k+1]]
        idx_G = (np.linalg.norm(np.array(solver.bz["G_values"]) - (g2-g1) ,axis=1)).argmin()
        a = solver.overlaps[0,k,k+1,:,:]
        b =  solver.overlaps[idx_G,solver.bz["c3_indices"][k],\
                solver.bz["c3_indices"][k+1],:,:]
        c = np.linalg.norm(np.diag(solver.c3_eigenvalues[k]) @ \
            np.array( a)@ \
            np.diag(np.conjugate(solver.c3_eigenvalues[k+1])) -b )
        print("overlap c3 invar:", c)
        print("g : of first is", solver.bz["c3_indices_of_Gs"][k])
        print("g : of second is:", solver.bz["c3_indices_of_Gs"][k+1])


    
    print("SHOWING DECAY OF OVERLAPS")
    for g in range(len(solver.bz["G_values"])):
        print("\n")
        print("Norm of G",np.linalg.norm(solver.bz["G_values"][g]))
        print("Overlap of k with k at G:",solver.overlaps[g,0,0,0,0])
    solver.check_v_c2t_invariance()

    plt.xticks(brillouin_zone["ticks_coords"],brillouin_zone["ticks_vals"])
    plt.legend()
    plt.show()



