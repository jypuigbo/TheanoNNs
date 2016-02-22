import scipy.signal as sg

import numpy as np
import matplotlib.pyplot as plt
import math
import random


import scipy.ndimage.filters as fi

import scipy.sparse as sp



def non_square_diagonal(s1,s2,sg=0.01, show_image=False):
	if s1 == s2:
		res = np.identity(s1)
	else: 
	    sigma=sg
	    mu = (np.arange(s1)+0.5)/s1

	    aux1 = (1/(sigma*np.sqrt(2*np.pi)))
	    aux2 = (1-np.arange(s2).reshape(s2,1)*np.ones([s2,s1])).transpose()/s2
	    aux2 = np.ones((s1,s2))* (np.arange(s2)+0.5)/s2
	    aux3 = np.ones([s2,s1])*mu

	    res = aux1*np.exp(-np.power(aux2-aux3.transpose(),2)/(np.power(sigma,2)))/float(2*s1)
	    
	    res -=np.min(res,1).reshape(res.shape[0],1)
	    res /= np.max(res,1).reshape(res.shape[0],1)
	    res *= res*res*res
	    res[res<0.6]=0
	    res -=np.min(res,1).reshape(res.shape[0],1)
	    res /= np.max(res,1).reshape(res.shape[0],1)
	    #res = res/np.sum(res,1).reshape(s1,1)

	if show_image:
	        plt.imshow(res)
	        plt.show()
	return res

def flattenKernel(k,n):
	'''Flattens a kernel into a line of a specific size
	k: kernel
	n: row length of the matrix
	output: new 1-dimensional kernel
	'''
	s = k.shape[0]
	new_kernel = np.zeros((s,n))
	print s/2.
	print n/2.
	print 'here!!!!!'
	new_kernel[int(s/2),int(n/2)]=1
	new_kernel = sg.convolve(new_kernel,k,mode='same')
	new_kernel = new_kernel.flatten(0)
	#new_kernel = k.flatten(0)
	#plt.imshow(k,aspect='auto')
	#plt.show()
	if True:
		plt.imshow(new_kernel.reshape(new_kernel.size,1),aspect='auto')
		plt.show()
	'''plt.imshow(k)
	plt.show()
	plt.imshow(new_kernel.reshape(1,new_kernel.size))
	plt.show()'''
	return new_kernel.reshape(new_kernel.size,1)

def fitMatrixSize(mat,shape1, shape2):
	'''After using convolutions to get the connection matrix, the matrix 
	becomes larger. This function makes it the same size as the number 
	of neurons'''
	#print 'fitting matrix'
	size = shape2[0]*shape2[1]
	if mat.shape[1] != size:
		print size
		#print "I'm in"
		m = size/2
		print m
		center = int(mat.shape[1]/2)
		if m < size/2.:
			print "UP!!!!!!!!!!!!!!!!!!!!"
			n_mat = mat[:,(center-int(m)):(center+int(m)+1)]
		else:
			print "Down!!!!!!!!!!!!!!!!!!1"
			n_mat = mat[:,(center-int(m)):(center+int(m))]
		print "[info] Shape was fit from " + str(mat.shape) + " to " + str(n_mat.shape)
	else:
		n_mat=mat
	
	size = shape1[0]*shape1[1]
	if mat.shape[0] != size:
		m = size/2
		center = int(mat.shape[0]/2)
		if m < size/2.:
			print "updown!!"
			n_mat = n_mat[(center-int(m)):(center+int(m)+1),:]
		else:
			
			n_mat = n_mat[(center-int(m)):(center+int(m)),:]
		print "[info] Shape was fit from " + str(mat.shape) + " to " + str(n_mat.shape)
	if (mat.shape[0] == shape1[0]*shape1[1]) and (mat.shape[1] == shape2[0]*shape2[1]):
		n_mat=mat
	return n_mat
	

def kernel2connection(k,inp, out,show_image=True):
	'''Converts a kernel based system into a connectivity matrix'''
	connections = non_square_diagonal(inp[0]*inp[1],out[0]*out[1], sg=0.1) # np.identity(n_row[0]*n_col[1])
	print "[info] Kernel shape: " + str(k.shape)
	print "[info] Connection Matrix shape: " + str(connections.shape)
	#print flattenKernel(k,n_col).shape
	aux = flattenKernel(k,out[1])
	#plt.imshow(aux, aspect='auto')
	#plt.show()
	connections = sg.convolve(connections,aux)
	if show_image:
		#print connections[25,:].shape
		plt.imshow(connections,aspect='auto')
		plt.show()
	connections = fitMatrixSize(connections,inp,out)
	if show_image:
		#print connections[25,:].shape
		plt.imshow(connections,aspect='auto')
		plt.show()
	
	return connections.astype('float32')
	# The matrix could be reordered for visualization (or speed?) using the following:
	# a breadth-first search. Since the reverse Cuthill-McKee ordering of a matrix


def gkern1(kern_shape, nsig=3,show_image=True):
    """Returns a 2D Gaussian kernel array. 
    size is kernlen, gaussian is centered 
    in the middle and std is 'nsig' units"""

    # create nxn zeros
    inp = np.zeros((kern_shape[0], kern_shape[1]))
    # set element at the middle to one, a dirac delta
    inp[kern_shape[0]//2, kern_shape[1]//2] = 1
    # gaussian-smooth the dirac, resulting in a gaussian filter mask
    aux =  fi.gaussian_filter(inp, nsig/2)
    #print np.max(aux)
    aux = aux/np.max(aux)
    #aux[aux<0.0000005]=0
    print "Remember to always check how is the kernel!!!"
    if show_image:
        plt.imshow(aux)
        plt.show()
    return aux

def gkern2(kern_shape, sigma, show_image=True):
    """Returns a 2D Gaussian kernel array. 
    size is kernlen, gaussian is centered 
    in the middle and std is 'nsig' units"""

    # create nxn zeros
    inp = np.zeros((kern_shape[0], kern_shape[0]))
    # set element at the middle to one, a dirac delta
    inp[kern_shape[0]//2, kern_shape[1]//2] = 1
    # gaussian-smooth the dirac, resulting in a gaussian filter mask
    aux =  fi.gaussian_filter(inp, sigma[0]/2)
    aux*=aux
    aux = aux/np.max(aux)
    aux1 = aux[:,kern_shape[0]//2]
    aux =  fi.gaussian_filter(inp, sigma[1]/2)
    aux = aux/np.max(aux)
    aux2 = aux[kern_shape[0]//2,:]
    #print np.max(aux)
    aux = sg.convolve(aux1.reshape(1,kern_shape[0]),aux2.reshape(kern_shape[0],1))
    
    #aux[aux<0.1]=0
    print aux.shape
    print "Remember to always check how is the kernel!!!"
    if show_image:
        plt.imshow(aux)
        plt.title('genKern ' + str(kern_shape))
        plt.show()
    return aux
'''def gkern3(kern_shape, sig1):
	return gkern3(kern_shape, sig1, sig1)
'''