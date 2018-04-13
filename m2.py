import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import time
from numpy.linalg import inv
import math

def main():

	# Reading data
	X = np.genfromtxt('weightedX.csv', delimiter='\n').reshape(100,1)
	X = X.transpose()
	Y = np.genfromtxt('weightedY.csv', delimiter='\n').reshape(100,1)
	X = np.vstack([X, X[0,:]])
	X[0,:] = 1.0
	X = X.transpose()

	## Data Preprocessing
	mean = np.mean(X[:,1]);
	var = np.var(X[:,1]);
	X[:,1] = (X[:,1]-mean)/math.sqrt(var);

	## Calculating value of theta
	Z1 = np.dot(X.transpose(),Y)
	Z2 = np.dot(X.transpose(), X)
	Z2 = inv(Z2)
	Theta = np.dot(Z2,Z1)

	## Plotting data and linear fit
	plt.plot(X[:,1].transpose(), Y, 'ro')
	prediction = np.dot(Theta.transpose(), X.transpose())
	plt.plot(X[:,1].transpose(), prediction.transpose(), label='Fit for unweighted linear regression')
	plt.xlabel('X')
	plt.ylabel('Y')
	plt.title('Unweighted Linear Regression')
	legend = plt.legend(loc='lower right')
	plt.show()	

if __name__ == "__main__":
    main()