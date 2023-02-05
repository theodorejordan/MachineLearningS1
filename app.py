import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_boston

boston = load_boston()
"""
X = boston.data[:, [5]].T
Y = boston.target
    
print(X)
print(Y)
"""


from GradientDescent import GradientDescent
from LDA import LDA

def normalize(data):
    mean = np.mean(data)
    std = np.std(data)
    return (data - mean) / std

def testGradientDescent():
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
    
    Y_pred = theta * X + b


def testGradientDescentBoston():

    X = boston.data[:, [5]].T
    Y = boston.target
    
    X = X[0]
    
    print(X)
    print("")
    print(Y)
    
    gd = GradientDescent()
    b, theta = gd.gradient_descent(X, Y, rate=0.01, num_iterations=2000, threshold = 0.00000000001, show = True)
    
    Y_pred = theta * X + b
    print(theta, " ", b) 
    
    plt.scatter(X, Y)
    plt.plot(X, Y_pred, color='red')
    plt.show()
    
  

def testLDA():
    # Training Set
    mean1 = [0, 0]
    cov1 = [[1, 0], [0, 1]]
    class1 = np.random.multivariate_normal(mean1, cov1, 100)

    mean2 = [6, 6]
    cov2 = [[1, 0], [0, 1]]
    class2 = np.random.multivariate_normal(mean2, cov2, 100)
    
    X_train = np.concatenate((class1, class2), axis=0)
    y_train = np.concatenate((np.zeros(100), np.ones(100)))
    
    # Testing Set
    mean1 = [3, 3]
    cov1 = [[1, 0], [0, 1]]
    classTest = np.random.multivariate_normal(mean1, cov1, 50)
    
    lda = LDA()
    lda.fit(X_train[:100], X_train[100:])
    
    y_test = lda.testClass(classTest)
        
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='rainbow')
    plt.scatter(classTest[:, 0], classTest[:, 1], c=y_test, cmap='rainbow', marker='x')

    plt.show()
    
def main():
    # Load the data from a file into a pandas dataframe
    df = pd.read_csv("heart.csv")
    
    label_encoder = LabelEncoder()
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = label_encoder.fit_transform(df[col])

    # Split the data into features (X) and target labels (y)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    # Train the LDA model on the data
    clf = LinearDiscriminantAnalysis()
    clf.fit(X, y)

    # Make predictions for all the data points
    predictions = clf.predict(X)

    correct_predictions = 0
    for i in range(len(y)):
        if y[i] == predictions[i]:
            correct_predictions += 1

    accuracy = correct_predictions / len(y)

    # Print the accuracy
    print("Accuracy:", accuracy)
    
    
    testGradientDescentBoston()
    
    
if __name__=="__main__":
    main()