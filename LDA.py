import numpy as np
import math

class LDA:
    def __init__(self, class0, class1):
        self.data0 = class0
        self.data1 = class1
        
    def getClass(self, x):
        inv_cov = np.linalg.inv(self.sumClass)
        
        omega0 = np.dot(np.dot(x, inv_cov), self.mu0) - 1 / 2 * np.dot(np.transpose(self.mu0), np.dot(self.mu0, inv_cov)) + math.log(self.pi0)
        omega1 = np.dot(np.dot(x, inv_cov), self.mu1) - 1 / 2 * np.dot(np.transpose(self.mu1), np.dot(self.mu1, inv_cov)) + math.log(self.pi1)

        if(max(omega0, omega1) == omega0):
            return 0
        
        return 1
    
    def generate_parameters(self):
        self.pi0 = len(self.data0) / (len(self.data0) + len(self.data1))
        self.pi1 = len(self.data1) / (len(self.data0) + len(self.data1))
        
        self.mu0 = np.mean(self.data0, axis=0)
        self.mu1 = np.mean(self.data1, axis=0)
        
        sum0 = 0;
        sum1 = 0;

        for i in range(len(self.data0)):
            sum0 += np.outer(self.data0[i] - self.mu0, np.transpose(self.data0[i] - self.mu0))
            
        for i in range(len(self.data1)):
            sum1 += np.outer(self.data1[i] - self.mu1, np.transpose(self.data1[i] - self.mu1))
        
        self.sumClass = (sum0 + sum1) / (len(self.data0) + len(self.data1))