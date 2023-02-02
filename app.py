import numpy as np
import matplotlib.pyplot as plt

from numpy.random import MT19937, RandomState, SeedSequence
from GradientDescent import GradientDescent
from LDA import LDA

def normalize(data):
    mean = np.mean(data)
    std = np.std(data)
    return (data - mean) / std

def rotate_cov(A, theta):
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    return R @ A @ R.T

def make_gaussian_blobs(means, covs, Ns = 100, seed = 42):
    ''' Create different blobs, each blob being a gaussian 
        from given mean and covariance
    '''
    rs = RandomState(MT19937(SeedSequence(seed)))
    M = min(len(means), len(covs))

    if not isinstance(Ns, list):
        aux = [Ns for _ in range(M)]
        Ns = aux
    X = np.concatenate([rs.multivariate_normal(m,
                                               cov = S,
                                               size = N)
                        for m, S, N in zip(means, covs, Ns)
                        ],
                       axis=0)
    Y = []

    for i, N in enumerate(Ns):
        Y += [i]*N

    Y = np.array(Y).ravel()

    return X, Y

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
    
    gd = GradientDescent()
    b, theta = gd.gradient_descent(X, Y, rate=0.01, num_iterations=2000, threshold = 0.00000000001, show = False)
    
    # Y_pred = theta * X + b
    # print(theta, " ", b)
    
    np.random.seed(1)

    # Generate 150 samples of 4-dimensional data
    data = np.random.randn(150, 2)
    
    # print(data)
    
    means = np.array([[0, 2], [1, -1]])
    covs = [rotate_cov(np.array([[1, 0.9],
                                [0.9, 1]]),np.pi/180*30),
            rotate_cov(np.array([[1, 0.2],
                                [0.2, 1]]),np.pi/180*0)
    ]
    X, Y = make_gaussian_blobs(means, covs, Ns = [100, 100], seed = 42)

    fig, ax = plt.subplots(1,1, figsize = (7,7))
    ax.scatter(X[:,0], X[:,1], c = Y)
    
    # print(X[:100])
    # print(len(X[:100, 0]))
    # print(X[:,1][50:])
    print(Y)
    testPoint = [4, -2]
    
    lda = LDA(X[:100], X[100:], len(X))
    lda.generate_parameters()
    cl = lda.getClass(testPoint)
    print(cl)
    
    ax.scatter(testPoint[0], testPoint[1], color = 'yellow' if cl == 1 else 'purple')
    plt.show()
    
if __name__=="__main__":
    main()