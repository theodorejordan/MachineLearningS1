import numpy as np
import matplotlib.pyplot as plt
import random

def normalize(data):
    mean = np.mean(data)
    std = np.std(data)
    return (data - mean) / std

def mean_squared_error(y_true, y_predicted):
    mse = np.sum((y_true-y_predicted)**2) / len(y_true)
    return mse

def update_t_b(X, Y, Y_hat, theta_0, b_0, rate):
    n = float(len(Y))
    
    derivative_t = (2 / n) * np.dot((Y_hat - Y), X)
    derivative_b = (2 / n) * np.sum(Y_hat - Y)
    
    theta_1 = theta_0 - rate * derivative_t
    b_1 = b_0 - rate * derivative_b
    
    return theta_1, b_1

def gradient_descent(X, Y, rate, num_iterations, threshold):
    theta, b = random.random(), random.random()
    iter = 0
    prev_cost = None
    
    iterations = []
    
    for i in range(num_iterations):
        pred = np.dot(X, theta) + b
        cost = mean_squared_error(Y, pred)
        
        if prev_cost and abs(prev_cost - cost) <= threshold:
            break
        
        prev_cost = cost
        
        theta, b = update_t_b(X, Y, pred, theta, b, rate)
        
        if(i%10==0):
            iterations.append([i, cost])
        
        iter += 1
        
    iter = [x for x,_ in iterations]
    costs = [y for _,y in iterations]
    plt.plot(iter, costs)
    plt.ylabel("Cost or MSE")
    plt.xlabel("Iterations")
    plt.show()
    return b, theta

def main():
    X = np.array([32.50234527, 53.42680403, 61.53035803, 47.47563963, 59.81320787,
           55.14218841, 52.21179669, 39.29956669, 48.10504169, 52.55001444,
           45.41973014, 54.35163488, 44.1640495 , 58.16847072, 56.72720806,
           48.95588857, 44.68719623, 60.29732685, 45.61864377, 38.81681754])
    
    Y = np.array([31.70700585, 68.77759598, 62.5623823 , 71.54663223, 87.23092513,
           78.21151827, 79.64197305, 59.17148932, 75.3312423 , 71.30087989,
           55.16567715, 82.47884676, 62.00892325, 75.39287043, 81.43619216,
           60.72360244, 82.89250373, 97.37989686, 48.84715332, 56.87721319])
    
    X = normalize(X)
    Y = normalize(Y)
    
    b, theta = gradient_descent(X, Y, rate=0.01, num_iterations=2000, threshold = 0.00000000001)
    
    Y_pred = theta * X + b
    
if __name__=="__main__":
    main()