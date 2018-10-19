import numpy as np
import matplotlib.pyplot as plt

#Loading the dataset
data = np.loadtxt('C:\Users\Dell\Desktop\Coursera ML\Assignments\machine-learning-ex2\ex2\ex2data1.txt',delimiter=",")
x = data[:, 0:2]
y = np.c_[data[:, 2]]

X = np.c_[np.ones((x.shape[0],1)),x]
m = X.shape[0]
n = X.shape[1]

theta = np.zeros((n,y.shape[1]))
#theta=np.zeros((n,1))
iter=1500
alpha=0.01

# Part 1 : Plotting

def plotData(X,y):
    #plt.plot(X[:, 0], X[:, 1], marker='o', c=y)
    #plt.plot(X[:, 1], marker='o', c=y)
    #plt.scatter(X[:, 0], marker='+', markeredgecolor='black', X[:, 1], marker='o', markeredgecolor='yellow')
    #plt.scatter(X[:, 1], marker='o', markeredgecolor='yellow')
    #plt.plot(x[:,0], 'b+', x[:,1], 'yo')
    #plt.scatter(x[:, 0], marker='+', c=y)
    #plt.scatter(x[:, 1], marker='o', c=b)
    #plt.scatter(X[:, 1], X[:, 2], marker='+', c=b)

    #plt.scatter(X[:, 1], X[:, 2], marker='o', c=y)
    pos = X[np.where(y == 1, True, False).flatten()]
    neg = X[np.where(y == 0, True, False).flatten()]
    plt.plot(pos[:, 1], pos[:, 2], 'k+', label='Admitted')
    plt.plot(neg[:, 1], neg[:, 2], 'yo', label='Not admitted')
    plt.xlabel('Exam 1 Score')
    plt.ylabel('Exam 2 Score')
    plt.title('Scatter Plot of Training Data')
    plt.legend()
    plt.show()

plotData(X,y)

#Part 2 : Compute Cost and Gradient

def sigmoid(z):
    g = 1/(1 + np.exp(-z))
    return g

def costFunction(theta, X, y):
    grad = np.zeros((theta.shape[0],1))
    htheta = sigmoid( X.dot(theta))
    J = np.sum((-y*np.log(htheta))-(((1-y)*np.log(1-htheta))))/m
    return J

C = costFunction(theta, X, y)
print "Initial Cost is : ",C

def gradientDescent(X,y,theta):
     #weights=np.ones((theta.shape[0],theta.shape[1]))

     """grad=np.zeros((theta.shape[0],1))
     for i in range(iter):
         htheta = sigmoid(X.dot(theta));
         #temp = (alpha/m) * np.sum(X.T.dot(htheta - y))
         grad = grad - ((alpha/m)*(X.T.dot(htheta - y)))
     return grad"""

     for i in range(iter):
         htheta=sigmoid(X.dot(theta));
         theta=theta-((alpha/m)*X.T.dot(htheta-y))
     return theta

g = gradientDescent(X,y,theta)
print "Gradient descent at test theta : ",g

