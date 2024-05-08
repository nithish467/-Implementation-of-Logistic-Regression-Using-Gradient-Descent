# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Use the standard libraries in python for finding linear regression.
2. Set variables for assigning dataset values.
3. Import linear regression from sklearn.
4. Predict the values of array.
5. Calculate the accuracy, confusion and classification report b importing the required modules from sklearn.
6. Obtain the graph.

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: NITHISH KUMAR S
RegisterNumber: 212223240109
*/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

data = np.loadtxt("ex2data1.txt",delimiter=",")
X = data[:,[0,1]]
Y = data[:,2]

X[:5]

Y[:5]

# VISUALIZING THE DATA
plt.figure()
plt.scatter(X[Y== 1][:, 0], X[Y==1][:,1],label="Admitted")
plt.scatter(X[Y==0][:,0],X[Y==0][:,1],label="Not admitted")
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.legend()
plt.show()

def sigmoid(z):
    return 1/(1+np.exp(-z))

plt.plot()
X_plot=np.linspace(-10,10,100)
plt.plot(X_plot,sigmoid(X_plot))
plt.show()

def costFunction(theta, X, Y):
    h = sigmoid(np.dot(X, theta))
    J = -(np.dot(Y, np.log(h)) + np.dot(1-Y,np.log(1-h))) / X.shape[0]
    grad = np.dot(X.T, h-Y)/X.shape[0]
    return J,grad

X_train = np.hstack((np.ones((X.shape[0],1)),X))
theta = np.array([0,0,0])
J,grad = costFunction(theta,X_train,Y)
print(J)
print(grad)

X_train = np.hstack((np.ones((X.shape[0],1)),X))
theta = np.array([-24,0.2,0.2])
J,grad = costFunction(theta,X_train,Y)
print(J)
print(grad)

def cost(theta,X,Y):
  h=sigmoid(np.dot(X,theta))
  J=-(np.dot(Y,np.log(h))+np.dot(1-Y,np.log(1-h)))/X.shape[0]
  return J

def gradient(theta,X,Y):
  h=sigmoid(np.dot(X,theta))
  grad=np.dot(X.T,h-Y)/X.shape[0]
  return grad

X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
res=optimize.minimize(fun=cost,x0=theta,args=(X_train,Y),method='Newton-CG',jac=gradient)
print(res.fun)
print(res.x)

def plotDecisionBoundary(theta,X,Y):
    X_min , X_max = X[:, 0].min() - 1,X[:,0].max() + 1
    Y_min , Y_max = X[:, 1].min() - 1,X[:,1].max() + 1
    XX,YY = np.meshgrid(np.arange(X_min,X_max,0.1),
                        np.arange(Y_min,Y_max,0.1))
    X_plot = np.c_[XX.ravel(), YY.ravel()]
    X_plot = np.hsatck((np.ones((X_plot.shape[0],1)),X_plot))
    Y_plot = np.dot(X_plot, theta).reshape(XX.shape)
    plt.figure()
    plt.scatter(X[Y==1][:,0],X[Y==1][:,1],label='Admitted')
    plt.scatter(X[Y==1][:,0],X[Y==1][:,1],label='Not admitted')
    plt.contour(XX,YY,Y_plot,levels=[0])
    plt.Xlabel("Exam 1 score")
    plt.Ylabel("Exam 2 score")
    plt.legend()
    plt.show()

print("Decision boundary-graph for exam score:")
plotDecisionBoundary(res.x,X,Y)


prob=sigmoid(np.dot(np.array([1,45,85]),res.x))
print(prob)

def predict(theta, X):
  X_train=np.hstack((np.ones((X.shape[0],1)),X))
  prob=sigmoid(np.dot(X_train,theta))
  return (prob >= 0.5).astype(int)

np.mean(predict(res.x,X)==y)


```

## Output:
### Array value of X:

![Screenshot 2024-05-08 182419](https://github.com/nithish467/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/150232274/1323efe4-57d9-4553-93e5-8fa90ff8d8aa)


### Array value of Y:
![Screenshot 2024-05-08 182432](https://github.com/nithish467/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/150232274/d1708bb6-b204-4e16-b107-dac18943a2ee)


### Exam 1-Score graph:

![Screenshot 2024-05-08 182444](https://github.com/nithish467/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/150232274/003d12a1-97ab-4fc1-9139-8d93574960ee)

### Sigmoid function graph:
![Screenshot 2024-05-08 182500](https://github.com/nithish467/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/150232274/6f44d5dd-afb8-42d0-83e9-87ebbb1a47a1)


### X_Train_grad value:
![Screenshot 2024-05-08 182515](https://github.com/nithish467/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/150232274/8a5ddce2-78bc-4f66-b525-0004b8886bee)



### Y_Train_grad value:
![Screenshot 2024-05-08 182531](https://github.com/nithish467/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/150232274/85807d21-5d2d-4808-bd1d-435fbfb85d7b)


### Print res.X:
![Screenshot 2024-05-08 182542](https://github.com/nithish467/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/150232274/9a90cb25-5397-42fc-b0c5-3fbeae90a754)


### Decision boundary-gragh for exam score:
![Screenshot 2024-05-08 182601](https://github.com/nithish467/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/150232274/2015c497-b5ec-4776-9753-4144cf8ac173)


### Probability value:
![Screenshot 2024-05-08 182611](https://github.com/nithish467/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/150232274/c7c8997d-dbe8-4acf-b4c5-edc945aef7d6)


### Prediction value of mean:
![Screenshot 2024-05-08 182646](https://github.com/nithish467/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/150232274/a22a13eb-0c7c-40d8-9e8b-c10b81029950)


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

