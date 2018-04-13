import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import time
from numpy.linalg import inv
import math

def get_weight(X, i, data, tau):
	return math.exp(((data - X[i,1])**2)/(-2.0*tau*tau))

## function to get optimized parameter Theta 
def prediction(X, Y, W):
	Z = np.dot(X.transpose(), W)
	Zy = np.dot(Z,Y)
	Theta = np.dot(inv(np.dot(Z,X)), Zy)
	return Theta

def main():

	# Reading data
	X = np.genfromtxt('weightedX.csv', delimiter='\n').reshape(100,1)
	X = X.transpose()
	Y = np.genfromtxt('weightedY.csv', delimiter='\n').reshape(100,1)
	X = np.vstack([X, X[0,:]])
	X[0,:] = 1.0
	X = X.transpose()
	print(X.shape)
	print(Y.shape)

	##Data Preprocessing
	mean = np.mean(X[:,1]);
	var = np.var(X[:,1]);
	X[:,1] = (X[:,1]-mean)/math.sqrt(var);

	## Initialize tau parameter
	tau = 0.8

	Y_value = np.zeros((100,1), dtype = float)
	X_sorted = np.sort(X,axis=0)

	## Predicting values for given inputs
	for j in range(100):
		W = np.zeros((100,100), dtype=float)
		data = X_sorted[j,1]
		for i in range(100):
			W[i,i] = get_weight(X,i,data,tau)
		Theta = prediction(X,Y,W)
		value = Theta[0,0] + Theta[1,0]*data
		Y_value[j,0] = value

	# Ploting data and hypothesis
	plt.plot(X[:,1].transpose(), Y, 'ro')
	plt.plot(X_sorted[:,1].transpose(), Y_value, label='Fit for locally weighted linear regression')
	plt.xlabel('X')
	plt.ylabel('Y')
	plt.title('Locally Weighted Linear Regression (tau = 0.8)')
	legend = plt.legend(loc='lower right', fontsize='small')
	plt.show()	

if __name__ == "__main__":
    main()