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
        colors = ['red', 'blue', 'green',]
        
        iterations = []
        
        for rate, color in zip([rate, rate * 0.1, rate * 0.01], colors):
            theta, b = random.random(), random.random()
            iter = 0
            prev_cost = None
            for i in range(num_iterations):
                pred = np.dot(X, theta) + b
                cost = self.mean_squared_error(Y, pred)
                
                if prev_cost and abs(prev_cost - cost) <= threshold:
                    break
                
                prev_cost = cost
                
                theta, b = self.update_t_b(X, Y, pred, theta, b, rate)
                
                if(i % 10 == 0):
                    iterations.append([i, cost, rate])
                
                iter += 1
                
            if(show):
                iter = [x for x,_,z in iterations if z==rate]
                costs = [y for _,y,z in iterations if z==rate]
                plt.plot(iter, costs, color=color, label=f'rate = {rate}')
            else:
                return b, theta
        
        plt.ylabel("Cost or MSE")
        plt.xlabel("Iterations")
        plt.legend()
        plt.show()
            
        return b, theta