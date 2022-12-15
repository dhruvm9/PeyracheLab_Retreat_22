#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 17:29:17 2022

@author: dl2820
"""

import numpy as np
import matplotlib.pyplot as plt


traindata = np.vstack([
    np.random.multivariate_normal([1,1], np.eye(2)*0.1, 100),
    np.random.multivariate_normal([-1,-1], np.eye(2)*0.1, 100),
    ])

testdata = np.vstack([
    np.random.multivariate_normal([1,1], np.eye(2)*1, 100),
    np.random.multivariate_normal([-1,-1], np.eye(2)*1, 100),
    ])

labels = np.hstack([np.zeros(100), np.ones(100)])


alpha = 0.01

def PerceptronError(data,labels,weights):
    activation = sigmoid(np.matmul(weights,data.T))
    meanError = np.mean(np.abs(labels-activation))
    return meanError

def sigmoid(x):
    return 1.0/(1 + np.exp(-x))

def PerceptronTrain(Data,MaxIter=100,errorThreshold=0.02):
    

    #initalize weights
    weights = np.random.randn(np.size(traindata,1))
    
    #initalize bias
    #bias = 0
    
    #MaxIter = 1000
    #errorThreshold = 0.01
    meanError = PerceptronError(Data,labels,weights)
    
    errortracker = []
    
    iter = 0
    while meanError>errorThreshold and iter<MaxIter:
        
        #loop through the whlole dataset, for each datapoint
        for dataIdx,datapoint in enumerate(Data):    
            #compute activation (classified category)
            activation = sigmoid(np.dot(weights,datapoint))
            
            #update the weights, based on error for that datapoint
            weights = weights + alpha*(labels[dataIdx]-activation)*datapoint
            
        iter+=1
        
        #calcualte error for whole dataset
        meanError = PerceptronError(Data,labels,weights)
        errortracker.append(meanError)
            
    plt.figure()
    plt.plot(errortracker,'.-') 
    plt.xlabel('Iterations')
    plt.ylabel('Error')
    plt.show()       
    return weights,errortracker
            
      
def PerceptrionTest(testData,weights):
    activation = sigmoid(np.matmul(weights,testData.T))
    testError = np.mean(np.abs(labels-activation))
    
    xvals = np.linspace(-3,3,100)
    plt.figure()
    plt.scatter(testData[:,0],testData[:,1],c=activation)
    plt.plot(xvals,-weights[0]*xvals/weights[1],'k')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()
    
    return testError



def pcaTrain(trainData):
    covmat = np.cov(trainData.T)
    eigvals,eigvecs = np.linalg.eig(covmat)
    return eigvecs


def pcaClassifier(testData,eigvecs):

    projection = np.dot(testData, eigvecs)
    classification = projection[:,0]>0
    
    plt.figure()
    plt.scatter(testData[:,0],testData[:,1],c=classification)
    #plt.plot(xvals,-weights[0]*xvals/weights[1],'k')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()
    
    return projection
    
#%%
