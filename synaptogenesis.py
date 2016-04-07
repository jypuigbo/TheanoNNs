
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sg
from time import time
neurons = np.zeros((10,100))
x = np.arange(0,1,0.1).reshape(10,1)*np.ones((10,100))
y = np.ones((10,100))*np.arange(0,10,0.1).reshape(1,100)
x=x.reshape(10*100,1)
y=y.reshape(10*100,1)
print y.shape
connections = np.identity((neurons.size))
distances = np.zeros(connections.shape)

x = x*np.ones((10*100,10*100))
y = y*np.ones((10*100,10*100))

dx = x-x.transpose()
dy = y-y.transpose()

distances = dx**2+dy**2



dw = np.zeros(connections.shape)
dC = np.zeros(connections.shape)
k=1 #influence of distance
K=0.5  #Max probability
_distances = 1/(1+k*distances)
plt.imshow(_distances)
plt.show()
aux_kernel = np.ones((1,2*distances.shape[1]))
# Must consider that more connections for the neuron penalize the probability of connections
# New dendrites appear if there's the probability AND correlated activity
C=10
for i in range(10):
	X = np.random.rand(10*100,1)
	mask_C_0 = np.ma.masked_where(connections==0,connections)
	plt.imshow(X, aspect='auto')
	plt.title('X')
	plt.show()
	t=time()
	probabilities = K*np.dot(connections*_distances,_distances.transpose())*mask_C_0.mask # sg.convolve(K*connections*np.dot(_distances,_distances.transpose()), aux_kernel,'same')*mask_C_0.mask
	print time()-t
	plt.imshow(probabilities)
	plt.title('probabilities')
	plt.show()
	print X.shape
	print connections.shape
	dw = np.dot(X,X.transpose())-np.dot(X, np.dot(X.transpose(),0.1*np.ones((connections.shape[0],connections.shape[1]))) )# Oja, right now
	print np.maximum(dw)
	plt.imshow(dw)
	plt.title('dW')
	plt.show()
	new_dendrites = probabilities + C*dw - 1#probabilities - np.random.rand(connections.shape[0],connections.shape[1])
	plt.imshow(new_dendrites>0)
	plt.title('new Dendrites')
	plt.show()
	#print np.isnan(dw).any()
	#print np.isnan(new_dendrites).any()
	dC = (new_dendrites > 0)* dw
	#print dC
	# update
	#print (mask_C_0.mask*dC)
	#print (connections>0)*dw
	connections = connections + (mask_C_0.mask*dC) #+ (connections>0)*dw
	print 'connections: ' + str((mask_C_0.mask*dC).size)
	plt.imshow((mask_C_0.mask*dC),vmin=0)
	plt.title('current Connections')
	plt.show()
