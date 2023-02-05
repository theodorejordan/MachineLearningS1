import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_boston
from GradientDescent import GradientDescent
from LDA import LDA

def normalize(data):
    mean = np.mean(data)
    std = np.std(data)
    return (data - mean) / std

def gradient_descent(X, Y, show):
    X = normalize(X)
    Y = normalize(Y)
    
    gd = GradientDescent()
    b, theta = gd.gradient_descent(X, Y, rate=0.01, num_iterations=2000, threshold = 0.0000001, show = show)
    
    Y_pred = theta * X + b
    
    plt.scatter(X, Y)
    plt.plot(X, Y_pred, color='red')
    plt.show()

def LDA(X_train, y_train, X_test):
    classTest = X_test
    
    lda = LDA()
    lda.fit(X_train[:100], X_train[100:])
    
    y_test = lda.testClass(classTest)
        
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='rainbow')
    plt.scatter(classTest[:, 0], classTest[:, 1], c=y_test, cmap='rainbow', marker='x')

    plt.show()

def testGradientDescent(show):
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
    b, theta = gd.gradient_descent(X, Y, rate=0.01, num_iterations=2000, threshold = 0.0000000000001, show = show)
    
    Y_pred = theta * X + b
    
    plt.scatter(X, Y)
    plt.plot(X, Y_pred, color='red')
    plt.show()

def testGradientDescentBoston():
    boston = load_boston()

    X = boston.data[:, [5]].T
    Y = boston.target
    
    X = X[0]
    
    gd = GradientDescent()
    b, theta = gd.gradient_descent(X, Y, rate=0.001, num_iterations=2000, threshold = 0.0000001, show = True)
    
    Y_pred = theta * X + b
    
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
    
def testLDAWithCSV(csv_file):
    df = pd.read_csv(csv_file)
    
    #transform the data
    label_encoder = LabelEncoder()
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = label_encoder.fit_transform(df[col])

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    # train the model
    clf = LinearDiscriminantAnalysis()
    clf.fit(X, y)
    
    predictions = clf.predict(X)

    # computing accuracy
    correct_predictions = 0
    for i in range(len(y)):
        if y[i] == predictions[i]:
            correct_predictions += 1

    accuracy = correct_predictions / len(y)

    print("Accuracy:", accuracy)
    
def testKNNWithCSV(csv_file):
    df = pd.read_csv(csv_file)
    
    # formatting data
    label_encoder = LabelEncoder()
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = label_encoder.fit_transform(df[col])

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    # we're gonna show the evolution of the accuracy depending on 20 different value of k
    k_range = range(1, 50)

    # and on different value of size for the training set
    trainingSizes = [0.2, 0.30, 0.40, 0.50]
    accuracies = []

    for training_size in trainingSizes:
        temp_accuracy_scores = []

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=training_size, random_state=0)
        num_samples = int(training_size * X_train.shape[0])

        # going through every k number of neighbors
        for k in k_range:
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(X_train[:num_samples], y_train[:num_samples])
            temp_accuracy_scores.append(knn.score(X_test, y_test))

        accuracies.append(temp_accuracy_scores)

    for i, accuracy_score in enumerate(accuracies):
        plt.plot(k_range, accuracy_score, label='Training set size: {:.0%}'.format(trainingSizes[i]))

    plt.xlabel('Number of Neighbors')
    plt.ylabel('Accuracy')
    plt.title('Accuracy of k-NN classifier according to different training set sizes')
    plt.legend()
    plt.show()
    
def testRandomForestWithCsv(csv_file):
    df = pd.read_csv(csv_file)
    
    label_encoder = LabelEncoder()
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = label_encoder.fit_transform(df[col])

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    # We're gonna compare different size for the training sets
    sizes = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    accuracies = []

    for size in sizes:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = size)

        # creating the Random Forest Classifier
        clf = RandomForestClassifier()

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        accuracy = clf.score(X_test, y_test)
        accuracies.append(accuracy)

    plt.plot(sizes, accuracies)
    plt.xlabel("Test Size")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Test Size")
    plt.show()
    
def main():
   testRandomForestWithCsv("heart.csv")
    
if __name__=="__main__":
    main()