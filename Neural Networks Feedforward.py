import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit
from scipy.io import loadmat
import random

data=loadmat('C:\Users\Dell\Desktop\Coursera ML\Assignments\machine-learning-ex3\ex3\ex3data1.mat')
weights=loadmat('C:\Users\Dell\Desktop\Coursera ML\Assignments\machine-learning-ex3\ex3\ex3weights.mat')

input_layer_size  = 400  #20x20 Input Images of Digits
hidden_layer_size = 25   #25 hidden units
num_labels = 10          #10 labels, from 1 to 10

X=data['X']
X=np.c_[np.ones((X.shape[0],1)),X]
Y=data['y']
#X is 5000 images. Each image is a row. Each image has 400 pixels unrolled (20x20)
#y is a classification for each image. 1-10, where "10" is the handwritten "0"

m=X.shape[0]
n=X.shape[1]

#print m
#print n
theta1=weights['Theta1']
theta2=weights['Theta2']

def sigmoid(z):
    s=expit(z)
    return s

def forward_propagation(X,theta1,theta2):
    a1=X
    #print a1.shape[0],a1.shape[1]
    z1=a1.dot(theta1.T)
    h1=sigmoid(z1)
    a2 = np.c_[np.ones((X.shape[0], 1)), h1]
    #print a2.shape[0],a2.shape[1]
    z2=a2.dot(theta2.T)
    htheta=sigmoid(z2); #hypothesis
    #print htheta.shape[0],htheta.shape[1]
    return htheta

hyp = forward_propagation(X,theta1,theta2)
pred = np.argmax(hyp, axis=1)+1 #getting the predicted output
print Y.flatten()
print pred
print 'Training set accuracy: {} %'.format(np.mean(pred == Y.ravel()) * 100) #calculating accuracy
