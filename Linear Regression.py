import numpy as np
import matplotlib.pyplot as plt

#Part 1 : WarmUpExercise
def warmUpExercise():
    return(np.identity(5))

print warmUpExercise()

#Part 2 : Plotting

#Loading Data

data = np.loadtxt('C:\Users\Dell\Desktop\Coursera ML\Assignments\machine-learning-ex1\ex1\ex1data1.txt',
                      delimiter=',')
x = data[:, 0]
y = np.c_[data[:,1]]
def plotData():
    plt.plot(x, y, 'rx')
    plt.xlabel('Population of City in 10,000s')
    plt.ylabel('Profit in $10,000s')
    plt.title('Scatter Plot of Training Data')
    plt.show()

plotData()

#Part 3 : Cost and Gradient Descent

m=y.size
X=np.c_[np.ones(data.shape[0]),x]
n=x.size
iterations=1500
alpha=0.01
theta=np.zeros((2,1))

def computeCost(X,y,theta):
    m = y.size
    J = 0
    htheta = X.dot(theta) #hypothesis
    J = (np.sum(np.square(htheta-y)))/(2*m)
    return J

C = computeCost(X,y,theta)
print "Cost computed = ",C

def gradient_descent(X,y,theta,alpha,iterations):
    m=y.size
    J_history = np.zeros(iterations)
    for i in range(0,iterations):
        htheta = X.dot(theta)
        theta = theta - ((alpha/m)* X.T.dot(htheta-y))
        J=computeCost(X,y,theta)
        J_history[i]=J
    return theta,J_history

# theta for minimized cost J
theta , Cost = gradient_descent(X, y,theta,alpha,iterations)
print('theta: ',theta)

print("Plotting line of best fit")
weights=gradient_descent(X,y,theta,alpha,iterations)[0]
plt.plot(x,y,'rx')
weights=np.c_[weights]
eqn=X.dot(weights)#calculating y with weigths obtained from gradient descent
plt.plot(x,eqn,'-')
plt.show()

cost=gradient_descent(X,y,theta,alpha,iterations)[1]
plt.plot(cost,"-")
plt.ylabel('Cost Function')
plt.xlabel('Iterations')
plt.show()
