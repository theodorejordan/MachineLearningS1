import numpy as np
import math

class LDA:
    def __init__(self, class0, class1, length):
        self.data0 = class0
        self.data1 = class1
        self.n = length
        
    def getClass(self, x):
        omega0 = np.dot(np.dot(x, self.sumClass0), self.mu0) - 1 / 2 * np.dot(np.transpose(self.mu0), np.dot(x, self.sumClass0)) + math.log(self.pi0)
        omega1 = np.dot(np.dot(x, self.sumClass1), self.mu1) - 1 / 2 * np.dot(np.transpose(self.mu1), np.dot(x, self.sumClass1)) + math.log(self.pi1)
        
        print(omega0)
        print(omega1)
        
        if(max(omega0, omega1) == omega0):
            return 0
        
        return 1
        
    def generate_parameters(self):
        self.pi0 = len(self.data0) / self.n
        self.pi1 = len(self.data1) / self.n
        
        
        self.mu0 = np.dot(1 / len(self.data0), sum(self.data0))
        self.mu1 = np.dot(1 / len(self.data1), sum(self.data1))
        
        sum0 = 0;
        sum1 = 0;

        for i in range(len(self.data0)):
            sum0 += np.outer(self.data0[i] - self.mu0, np.transpose(self.data0[i] - self.mu0))
            
        for i in range(len(self.data1)):
            sum1 += np.outer(self.data1[i] - self.mu1, np.transpose(self.data1[i] - self.mu1))
        
        self.sumClass0 = np.dot(1 / len(self.data0), sum0)
        self.sumClass1 = np.dot(1 / len(self.data1), sum1)