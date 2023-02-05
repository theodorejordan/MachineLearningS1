import numpy as np
import matplotlib.pyplot as plt
import random

class GradientDescent:
    def mean_squared_error(self, y_true, y_predicted):
        mse = np.sum((y_true-y_predicted)**2) / len(y_true)
        return mse
    
    def update_t_b(self, X, Y, Y_hat, theta_0, b_0, rate):
        n = float(len(Y))
    
        derivative_t = (2 / n) * np.dot((Y_hat - Y), X)
        derivative_b = (2 / n) * np.sum(Y_hat - Y)
        
        theta_1 = theta_0 - rate * derivative_t
        b_1 = b_0 - rate * derivative_b
        
        return theta_1, b_1
    
    def gradient_descent(self, X, Y, rate, num_iterations, threshold, show):
        theta, b = random.random(), random.random()
        iter = 0
        prev_cost = None
        
        iterations = []
        
        for i in range(num_iterations):
            pred = np.dot(X, theta) + b
            cost = self.mean_squared_error(Y, pred)
            
            if prev_cost and abs(prev_cost - cost) <= threshold:
                break
            
            prev_cost = cost
            
            theta, b = self.update_t_b(X, Y, pred, theta, b, rate)
            
            if(i % 10 == 0):
                iterations.append([i, cost])
            
            iter += 1
            
        if(show):
            iter = [x for x,_ in iterations]
            costs = [y for _,y in iterations]
            plt.plot(iter, costs)
            plt.ylabel("Cost or MSE")
            plt.xlabel("Iterations")
            plt.show()
            
        return b, theta