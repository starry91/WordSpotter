import numpy as np
from sklearn.kernel_approximation import RBFSampler
from copy import deepcopy

class CCA():
    def __init__(self, n_components=192, ro=1e-4):
        self.output_dim = n_components
        self.ro = ro
    
    def find_w(self, Caa, Cab, Cbb, Cba):
        # print("In find W")
        Z = np.dot(np.linalg.inv(Caa), Cab)
        Z = np.dot(Z, np.linalg.inv(Cbb))
        Z = np.dot(Z, Cba)
        # print("After calculating Z")
        eigen_values, eigen_vectors = np.linalg.eigh(Z)
        eigen_values = np.absolute(eigen_values)
        ix = np.argsort(eigen_values)[::-1]
        eigen_vectors = eigen_vectors[:,ix]
        # print("After calculating eigen vectors")
        return eigen_vectors[:self.output_dim]
        
    ### shape of A Nxd
    ### shape of B Nxd
    def fit(self, A, B):
        # print("Inside Fit")
        A = deepcopy(A).T
        B = deepcopy(B).T
        # print("After deep copy")
        N = len(A)
        self.mu_a = np.mean(A, axis=1, keepdims=True)
        self.mu_b = np.mean(B, axis=1, keepdims=True)
        # print("After Mean")
        Caa = (1/N)*(np.dot((A-self.mu_a), (A-self.mu_a).T) + self.ro*np.eye(len(A)))
        Cbb = (1/N)*(np.dot((B-self.mu_b), (B-self.mu_b).T) + self.ro*np.eye(len(B)))
        Cab = (1/N)*(np.dot((A-self.mu_a), (B-self.mu_b).T))
        Cba = Cab.T
        # print("All C's calculated")
        self.project_a = self.find_w(Caa, Cab, Cbb, Cba)
        self.project_b = self.find_w(Cbb, Cba, Caa, Cab)
        # print("returning from fit")
        
    def transform_a(self, A):
        # print("In transform")
        A = deepcopy(A)
        return np.dot((A-self.mu_a.T), self.project_a.T)
    
    def transform_b(self, B):
        B = deepcopy(B)
        # print("In transform")
        return np.dot((B-self.mu_b.T), self.project_b.T)

class KCCA():
    def __init__(self, n_components=256):
        self.CCA = CCA(n_components)
    
    ### shape of A Nxd
    def fit(self, A, B):
        A = deepcopy(A)
        B = deepcopy(B)
        self.rbf_feature_A = RBFSampler(gamma=1, n_components=len(A))
        self.rbf_feature_B = RBFSampler(gamma=1, n_components=len(B))
        self.rbf_feature_A.fit(A)
        self.rbf_feature_B.fit(B)
        A = self.rbf_feature_A.transform(A)
        B = self.rbf_feature_B.transform(B)
        self.CCA.fit(A, B)
        
    def transform_a(self, A):
        A = deepcopy(A)
        A = self.rbf_feature_A.transform(A)
        return self.CCA.transform_a(A)
    
    def transform_b(self, B):
        B = deepcopy(B)
        B = self.rbf_feature_B.transform(B)
        return self.CCA.transform_b(B)