# Readme for Gradient Descent and LDA Implementation

This repository contains code for implementing gradient descent for linear regression and linear discriminant analysis (LDA) for classification. The code was written in the context of a machine learning course at Univeristy Paris Dauphine.

## Requirements

The following python libraries are required to run the code:

- Numpy
- Matplotlib
- Pandas
- Sklearn

## Gradient Descent

The GradientDescent class implements the gradient descent algorithm for linear regression. The gradient_descent method calculates the parameters b and theta that minimize the mean squared error between the actual and predicted target values. The method also provides the option to visualize the convergence of the algorithm by plotting the predicted target values against the actual target values.

The testGradientDescent method demonstrates how to use the GradientDescent class to fit a linear regression model on a simple dataset. The testGradientDescentBoston method shows how to use the GradientDescent class to fit a linear regression model on the Boston housing dataset from sklearn.

## LDA

The LDA class implements linear discriminant analysis for classification. The fit method trains the LDA model on the training data, and the testClass method uses the trained model to predict the class labels for a test dataset. The method also provides a visualization of the class boundaries and the predicted class labels for the test data.

The testLDA method demonstrates how to use the LDA class to classify a simple two-class dataset.

## Usage

To use the code, simply run the desired test method (e.g., testGradientDescent, testGradientDescentBoston, testLDA). The methods include all necessary imports and provide examples of how to use the implemented classes.

You should call the gradient_descent function with the right argument to use the gradient descent algorithm. And the LDA method to use the LDA algorithm.

Done by Julie Bachelet and Theodore Jordan