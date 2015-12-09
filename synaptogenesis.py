
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sg

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

#plt.imshow(distances)
#plt.show()

dw = np.zeros(connections.shape)
dC = np.zeros(connections.shape)
k=1 #influence of distance
K=0.5  #Max probability
_distances = 1/(1+distances)

aux_kernel = np.ones((1,2*distances.shape[1]))

for i in range(200):
	X = np.random.rand(10*100,1)
	mask_C_0 = np.ma.masked_where(connections==0,connections)
	plt.imshow(connections*_distances*_distances.transpose(),vmin=0, vmax=1)
	plt.show()
	probabilities = sg.convolve(K*connections*_distances*_distances.transpose(), aux_kernel,'same')*mask_C_0.mask

	plt.imshow(probabilities,vmin=0, vmax=1)
	plt.show()

	dw = np.dot(X,X.transpose()-np.dot(X, np.dot(X.transpose(),connections))) # Oja, right now
	new_dendrites = probabilities - np.random.rand(connections.shape)
	dC = np.ma.masked_where(new_dendrites < 0, dw)
	# update
	connections = connections + np.ma.masked_where(mask_C_0.mask, dC) + np.ma.masked_where(connections>0, dw)
