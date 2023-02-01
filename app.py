import numpy as np
import matplotlib.pyplot as plt

def mean_squared_error(y_true, y_predicted):
    mse = np.sum((y_true-y_predicted)**2) / len(y_true)
    return mse

def gradient_descent(x, y, iterations = 500, threshold = 0.001, rate = 0.01):
    cur_weight = 0.1
    cur_bias = 0.1
    prev_cost = None
    
    n = float(len(x))
    costs = []
    weights = []
    
    for i in range(iterations):
        y_pred = (cur_weight * x) + cur_bias;
        
        cur_cost = mean_squared_error(y, y_pred)
        
        if prev_cost and abs(prev_cost - cur_cost) <= threshold:
            break
        
        prev_cost = cur_cost
        
        costs.append(cur_cost)
        weights.append(cur_weight)
        
        # gradients
        weight_derivative = - (2 / n) * sum(x * (y - y_pred))
        bias_derivative = - (2 / n) * sum(y - y_pred)
        
        # update weight and bias
        cur_weight = cur_weight - (rate * weight_derivative)
        cur_bias = cur_bias - (rate * bias_derivative)
        
        print(f"Iteration {i+1}: Cost {cur_cost}, Weight \
        {cur_weight}, Bias {cur_bias}")
    
    plt.figure(figsize = (8,6))
    plt.plot(weights, costs)
    plt.scatter(weights, costs, marker='o', color='red')
    plt.title("Cost vs Weights")
    plt.ylabel("Cost")
    plt.xlabel("Weight")
    plt.show()
    
    return cur_weight, cur_bias

def main():
    X = np.array([32.50234527, 53.42680403, 61.53035803, 47.47563963, 59.81320787,
           55.14218841, 52.21179669, 39.29956669, 48.10504169, 52.55001444,
           45.41973014, 54.35163488, 44.1640495 , 58.16847072, 56.72720806,
           48.95588857, 44.68719623, 60.29732685, 45.61864377, 38.81681754])
    Y = np.array([31.70700585, 68.77759598, 62.5623823 , 71.54663223, 87.23092513,
           78.21151827, 79.64197305, 59.17148932, 75.3312423 , 71.30087989,
           55.16567715, 82.47884676, 62.00892325, 75.39287043, 81.43619216,
           60.72360244, 82.89250373, 97.37989686, 48.84715332, 56.87721319])
    
    rand = np.random.RandomState(19)
    w_init = rand.uniform(-10,10,2)
    
    est_weight, est_bias = gradient_descent(X, Y, iterations=500, threshold=0.000001, rate=0.0001)
    print(f"Estimated Weight: {est_weight}\nEstimated Bias: {est_bias}")
    
    Y_pred = est_weight*X + est_bias
 
    # Plotting the regression line
    plt.figure(figsize = (8,6))
    plt.scatter(X, Y, marker='o', color='red')
    plt.plot([min(X), max(X)], [min(Y_pred), max(Y_pred)], color='blue',markerfacecolor='red',
             markersize=10,linestyle='dashed')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()
    
if __name__=="__main__":
    main()