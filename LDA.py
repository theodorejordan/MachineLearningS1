import numpy as np
import math

class LDA:        
    def testClass(self, data):
        y_test = np.array(np.zeros(50))
        
        for i in range(len(data)):
            el = data[i]
            cl = self.getClass(el)
            y_test[i] = cl
            
        return y_test
    
    def getClass(self, x):
        inv_cov = np.linalg.inv(self.sumClass)
        
        omega0 = np.dot(np.dot(x, inv_cov), self.mu0) - 1 / 2 * np.dot(np.transpose(self.mu0), np.dot(self.mu0, inv_cov)) + math.log(self.pi0)
        omega1 = np.dot(np.dot(x, inv_cov), self.mu1) - 1 / 2 * np.dot(np.transpose(self.mu1), np.dot(self.mu1, inv_cov)) + math.log(self.pi1)

        if(max(omega0, omega1) == omega0):
            return 0
        
        return 1
    
    def fit(self, class0, class1):
        self.pi0 = len(class0) / (len(class0) + len(class1))
        self.pi1 = len(class1) / (len(class0) + len(class1))
        
        self.mu0 = np.mean(class0, axis=0)
        self.mu1 = np.mean(class1, axis=0)
        
        sum0 = 0;
        sum1 = 0;

        for i in range(len(class0)):
            sum0 += np.outer(class0[i] - self.mu0, np.transpose(class0[i] - self.mu0))
            
        for i in range(len(class1)):
            sum1 += np.outer(class1[i] - self.mu1, np.transpose(class1[i] - self.mu1))
        
        self.sumClass = (sum0 + sum1) / (len(class0) + len(class1))