# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Step 1: Start 
Step 2: Import modules
Step 3: Read the file
Step 4: Drop the serial no and salary columns
Step 5: Categorise columns for further labelling
Step 6: Label the columns
Step 7: Display dataset
Step 8: Select the features and labels
Step 9: Display dependent variables
Step 10: Initialize the model parameters.
Step 11: Define the sigmoid function.
Step 12: Define the loss function.
Step 13: Define the gradient descent algorithm.
Step 14: Train the model.
Step 15: Make predictions.
Step 16: Evaluate the model.
Step 17: Stop

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: Joel John Jobinse
RegisterNumber:  212223240062
*/
#import modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#reading the file
dataset=pd.read_csv("Placement_Data.csv")
dataset

#dropping the serial no and salary col
dataset=dataset.drop('sl_no',axis=1)
dataset=dataset.drop('salary',axis=1)

#categorising col for further labelling
dataset["gender"]=dataset["gender"].astype('category')
dataset["ssc_b"]=dataset["ssc_b"].astype('category')
dataset["hsc_b"]=dataset["hsc_b"].astype('category')
dataset["degree_t"]=dataset["degree_t"].astype('category')
dataset["workex"]=dataset["workex"].astype('category')
dataset["specialisation"]=dataset["specialisation"].astype('category')
dataset["status"]=dataset["status"].astype('category')
dataset["hsc_s"]=dataset["hsc_s"].astype('category')
dataset.dtypes

#labelling the columns
dataset["gender"]=dataset["gender"].cat.codes
dataset["ssc_b"]=dataset["ssc_b"].cat.codes
dataset["hsc_b"]=dataset["hsc_b"].cat.codes
dataset["degree_t"]=dataset["degree_t"].cat.codes
dataset["workex"]=dataset["workex"].cat.codes
dataset["specialisation"]=dataset["specialisation"].cat.codes
dataset["status"]=dataset["status"].cat.codes
dataset["hsc_s"]=dataset["hsc_s"].cat.codes

#display dataset
dataset

#selecting the features and labels
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,-1].values

#display dependednt variables
Y

#Initialize the model parameters.
theta=np.random.randn(X.shape[1])
y=Y
#Define the sigmoid function.
def sigmoid(z):
    return 1/(1+np.exp(-z))
#Define the loss function.
def loss(theta,X,y):
    h=sigmoid(X.dot(theta))
    return -np.sum(y*np.log(h)+(1-y)*np.log(1-h))

#Define the gradient descent algorithm.
def gradient_descent(theta, X, y, alpha, num_iterations):
    m=len(y)
    for i in range(num_iterations):
        h=sigmoid(X.dot(theta))
        gradient=X.T.dot(h-y)/m
        theta-=alpha*gradient
    return theta

#Train the model.
theta=gradient_descent(theta, X, y, alpha=0.01, num_iterations=1000)

#Make predictions.
def predict(theta, X):
    h=sigmoid(X.dot(theta))
    y_pred=np.where(h>=0.5,1,0)
    return y_pred

y_pred=predict(theta,X)

#Evaluate the model.
accuracy=np.mean(y_pred.flatten()==y)
print("Accuracy:",accuracy)

print(y_pred)

print(Y)

xnew=np.array([[0,87,0,95,0,2,78,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print(y_prednew)

xnew=np.array([[0,0,0,0,0,2,8,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print(y_prednew)
```

## Output:

![ml_exp5_output1](https://github.com/joeljohnjobinse/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/138955488/dc5d0088-e37a-4608-b7a2-1002d400da74)

![ml_exp5_output2](https://github.com/joeljohnjobinse/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/138955488/3088a79e-580c-486e-8edd-4d571b4f3d39)

![ml_exp5_output3](https://github.com/joeljohnjobinse/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/138955488/c10d448d-68dc-48c7-b586-282dd2d7ae3a)

![ml_exp5_output4](https://github.com/joeljohnjobinse/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/138955488/625d0f0e-0c5f-4b38-abcd-75ce983d333a)

![ml_exp5_output5](https://github.com/joeljohnjobinse/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/138955488/0641fae9-f932-4d08-a374-9ed4f1a3ee11)



## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

