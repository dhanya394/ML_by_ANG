import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('C:\Users\Dell\Desktop\Coursera ML\Assignments\machine-learning-ex2\ex2\ex2data2.txt', delimiter=',')

x = data[:, 0:2]
y = np.c_[data[:, 2]]

X = np.c_[np.ones((x.shape[0],1)),x]
m = X.shape[0]
n = X.shape[1]
theta = np.zeros((n,y.shape[1]))
iter=1500
alpha=0.01
l=1

def plotData(X,y):
    pos = X[np.where(y == 1, True, False).flatten()]
    neg = X[np.where(y == 0, True, False).flatten()]
    plt.plot(pos[:, 1], pos[:, 2], 'k+', label='Admitted')
    plt.plot(neg[:, 1], neg[:, 2], 'yo', label='Not admitted')
    plt.xlabel('Microchip Test 1')
    plt.ylabel('Microchip Test 2')
    plt.title('Scatter Plot of Training Data')
    plt.legend()
    plt.show()

plotData(X,y)

def sigmoid(z):
    g = 1/(1 + np.exp(-z))
    return g

def costFunctionReg(theta,X,y,l):
    htheta = sigmoid(X.dot(theta))
    J = np.sum((-y * np.log(htheta)) - (((1 - y) * np.log(1 - htheta))))/m
    J = J + ((l/(2*m))*np.sum(theta**2))
    return J

C = costFunctionReg(theta,X,y,l)
print "Cost after regularization : ",C

def regGradientDescent(theta,X,y,l):
    for i in range(iter):
        htheta = sigmoid(X.dot(theta))
        #htheta0=sigmoid(X.dot(theta[0,0]))
        theta[0,0] = theta[0,0] - ((alpha / m)*(((X.T.dot(htheta-y))[0,0])))
        theta[1:,0] = theta[1:,0] - ((alpha / m) * (X.T.dot(htheta - y))[1:,0]) - ((l/m)*(theta[1:,0]))
    return theta

"""htheta = sigmoid(X.dot(theta))
print htheta.shape[0],htheta.shape[1]
print y.shape[0],y.shape[1]
print theta.shape[0],theta.shape[1]
print X.shape[0],X.shape[1]"""

g = regGradientDescent(theta,X,y,l)
print "Gradient at test theta after regularization : ",g