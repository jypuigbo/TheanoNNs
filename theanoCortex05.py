# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

# Ask Pablo for emotion convNets (3d for motion, inhibition for better differentiation. 2 channels + cross-channels-> must be high level. Shunting inhibitory fields: x/max(x))
# 

import scipy.signal as sg
import sys
sys.path.append("/home/jy/robotology/yarp/bindings/build")
import yarp
import numpy as np
import matplotlib.pyplot as plt
import math
import random

import theano
from theano import tensor as T
from theano.tensor.nnet import conv
import scipy.ndimage.filters as fi
from theano import sparse
import scipy.sparse as sp

# Params

#theano.config.compute_test_value = 'warn'
theano.config.on_unused_input = 'warn'
theano.config.exception_verbosity='high'
theano.config.profile = False
theano.config.floatX='float32'
#theano.config.device = 'gpu'


def generateRandomSignalBut(epoch,center_F,size=512,f=(0,12),but=0,noi=0.333, n_harmonics=0):
    #f=(110,220,330,440,660,880)
    x= np.random.rand(size)*noi
    Nx=0
    if epoch%5 == 0:
        it=(epoch%(int(10000/30)))%len(f)
        center_F = np.random.choice(f,1)/2
        print "New CF: "  + str(center_F) + " from " + str(f)
        while center_F==but:
            center_F = np.random.choice(f,1)/2
    for i in range(3):
    	#print max(0,int(center_F)+i-4)
        x[max(0,int(center_F)+i-1)]=1
        Nx+=1
    SNR=pow(Nx,2)/pow(np.sum(x)-Nx,2)
    x[x<0.001]=0
    print x
    
    #print size
    #print "Shape: " + str(x.shape)
    #print "CF = " + str(center_F)
    return x,center_F,SNR


def non_square_diagonal(s1,s2,sg=0.01, show_image=False):
	#s1 = s1[0]*s1[1]
	#s2 = s2[0]*s2[1]
	if s1 == s2:
		return np.identity(s1)
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
	    res *= res
	    res = res/np.sum(res,1).reshape(s1,1)
	    if show_image:
	        plt.imshow(res)
	        plt.show()
	    plt.imshow(res)
	    return res

def flattenKernel(k,n):
	'''Flattens a kernel into a line of a specific size
	k: kernel
	n: row length of the matrix
	output: new 1-dimensional kernel
	'''
	s = k.shape[0]
	new_kernel = np.zeros((s,n))
	new_kernel[:,int(n/2)]=1
	new_kernel = sg.convolve(new_kernel,k)
	new_kernel = new_kernel.flatten(0)
	'''plt.imshow(k)
	plt.show()
	plt.imshow(new_kernel.reshape(1,new_kernel.size))
	plt.show()'''
	return new_kernel.reshape(1,new_kernel.size)

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
		print center
		print center-int(m)
		print center+int(m)
		if m < size/2.:
			n_mat = mat[:,center-int(m):center+int(m)+1]
		else:
			n_mat = mat[:,center-int(m):center+int(m)]
		print "[info] Shape was fit from " + str(mat.shape) + " to " + str(n_mat.shape)
	
	size = shape1[0]*shape1[1]
	if mat.shape[0] != size:
		#print "I'm in"
		m = size/2
		#print m
		center = int(mat.shape[0]/2+1)
		#print center
		if m < size/2.:
			n_mat = mat[center-int(m):center+int(m)+1,:]
		else:
			n_mat = mat[center-int(m):center+int(m),:]
		print "[info] Shape was fit from " + str(mat.shape) + " to " + str(n_mat.shape)
	if (mat.shape[0] == shape1[0]*shape1[1]) and (mat.shape[1] == shape2[0]*shape2[1]):
		n_mat=mat
	return n_mat
	

def kernel2connection(k,inp, out,show_image=False):
	'''Converts a kernel based system into a connectivity matrix'''
	connections = non_square_diagonal(inp[0]*inp[1],out[0]*out[1], sg=0.2) # np.identity(n_row[0]*n_col[1])
	print "[info] Kernel shape: " + str(k.shape)
	print "[info] Connection Matrix shape: " + str(connections.shape)
	#print flattenKernel(k,n_col).shape
	
	connections = sg.convolve(connections,flattenKernel(k,out[1]))
	if show_image:
		plt.imshow(connections)
		plt.show()
	connections = fitMatrixSize(connections,inp,out)
	if show_image:
		plt.imshow(connections)
		plt.show()
	
	return connections.astype('float32')
	# The matrix could be reordered for visualization (or speed?) using the following:
	# a breadth-first search. Since the reverse Cuthill-McKee ordering of a matrix


def gkern2(kern_shape, nsig=3,show_image=False):
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
    aux[aux<0.0005]=0
    print "Remember to always check how is the kernel!!!"
    if show_image:
        plt.imshow(aux)
        plt.show()
    return aux

#Define a constant random seed
#rng = np.random.RandomState()

# instantiate 4D tensor for input

#input = T.tensor4(name='input')
class connection(object):
	def __init__(self, input, filter_shape, sigma, weights_file):
		try: 
			if generate:
				np.load('asd')
			else:
				W=np.load(weights_file)
				print '[info] Weights loaded from file!'
				print 'Shape = ' + str(W.shape)
		except IOError:
			print "[info] Weights file wasn't found. Generating new connections"
			kern1 = gkern2(filter_shape,sigma)
			W = kernel2connection(kern1, i_shape, o_shape)
			#Wi /= np.sum(Wi,1).reshape((Wi.shape[0],1))*15
			print 'Shape = ' + str(Wi.shape)
			np.save(i_file,Wi)

class HebbianAdaptiveLayer(object):
	def __init__(self, input, filter_shape, sigma,i_shape,o_shape, i_file, r_file, Wi = False, Wr = False):
		global generate
		# Mean neuron density ~80k/mm^3 in V2 (skoglund 1996)
		# Synapse length follow a power law ()
		# Synapse length for feedback interareal ~10-40mm, feedforward same, but less connections
		# Synapse lengths is found by Sholl analysis. 
		# Ahould compare RF data with Van den Bergh 2010

		# Initialize weights as a shared variable
		#n_col=input.shape[1]

		try: 
			if generate:
				np.load('asd')
			else:
				Wi=np.load(i_file)
				print '[info] Weights loaded from file!'
				print 'Shape = ' + str(Wi.shape)
		except IOError:
			print "[info] Weights file wasn't found. Generating new connections"
			kern1 = gkern2(filter_shape,sigma)
			Wi = kernel2connection(kern1, i_shape, o_shape)
			#Wi /= np.sum(Wi,1).reshape((Wi.shape[0],1))*15
			print 'Shape = ' + str(Wi.shape)
			np.save(i_file,Wi)

		try: 
			if generate:
				np.load('asd')
			else:
				Wr=np.load(r_file)
				print 'Weights loaded from file!'
		except IOError:
			print "Weights file wasn't found. Generating new connections"
			kern2 = gkern2(filter_shape,sigma)
			Wr = kernel2connection(kern2, o_shape,o_shape)
			#Wr /= np.sum(Wi,1)
			np.save(r_file,Wr)

		if np.sum(Wi,1)[0] != 1:
			Wi /= np.sum(Wi,1).reshape((Wi.shape[0],1))*5
		if np.sum(Wr,1)[0] != 1:
			Wr /= np.sum(Wr,1).reshape((Wr.shape[0],1))
		print np.sum(Wi,0)
		print np.sum(Wi,1)
		plt.plot(Wi[1,:])
		plt.show()


		self.Wi= theano.shared( 
				sp.csc_matrix(
				np.asarray( 
				Wi, 
				dtype=input.dtype) ), name ='Wi')
		self.Wr = theano.shared( 
				sp.csc_matrix(
				np.asarray( 
				Wr, 
				dtype=input.dtype) ), name ='Wr')
		# Output of the layer is the sigmoid of the convolved network
		self.state = theano.shared( 
			sp.csc_matrix(
			np.asarray( 
			np.zeros((o_shape[0]*o_shape[1],1)), 
			dtype=input.dtype) ), name ='St')

		self.input = input

		# I could do the same with biases if needed
		#print self.input.get_value().shape
		#print self.Wi.get_value().shape
		self.output = theano.shared( 
			sp.csc_matrix(
			np.asarray( 
			np.zeros((o_shape[0]*o_shape[1],1)), 
			dtype=input.dtype) ), name ='Out')
		#sparse.structured_sigmoid(sparse.structured_dot(self.input, self.Wi))  #T.dot(self.input, self.Wi))
		# input = external + recursive (from layer)
		# self.input = T.dot(input, self.Wi) #+ T.sum(T.dot(self.state,self.Wr),1)

		# out: nx1
		# Wi: mxn
		# outT x WiT : 1xm
		self.yw = sparse.structured_dot(
						sparse.transpose(self.output),
						sparse.transpose(self.Wi))
		# in: nx1
		self.x_yw = sparse.sub(
						sparse.transpose(self.input),
						self.yw)


		# optional: self.output = T.nnet.sigmoid(conv_out+self.output)
		self.params = [self.Wi, self.Wr, self.state, self.output]
		#self.input=input



# de Prova
class HebbianInhibitoryLayer(object):
	def __init__(self, input, filter_shape, sigma,i_shape,o_shape, i_file, r_file, Wi = False, Wr = False):
		global generate
		# Mean neuron density ~80k/mm^3 in V2 (skoglund 1996)
		# Synapse length follow a power law ()
		# Synapse length for feedback interareal ~10-40mm, feedforward same, but less connections
		# Synapse lengths is found by Sholl analysis. 
		# Ahould compare RF data with Van den Bergh 2010

		# Initialize weights as a shared variable
		#n_col=input.shape[1]
		try: 
			if generate:
				np.load('asd')
			else:
				Wi=np.load(i_file)
				print '[info] Weights loaded from file!'
				print 'Shape = ' + str(Wi.shape)
		except IOError:
			print "[info] Weights file wasn't found. Generating new connections"
			kern1 = gkern2(filter_shape,sigma)
			kern1 = np.zeros(filter_shape)
			Wi = kernel2connection(kern1, o_shape, o_shape)
			plt.imshow(Wi)
			Wi = Wi
			#Wi /= np.sum(Wi,1).reshape((Wi.shape[0],1))*15
			print 'Shape = ' + str(Wi.shape)
			np.save(i_file,Wi)

		try: 
			if generate:
				np.load('asd')
			else:
				Wr=np.load(r_file)
				print 'Weights loaded from file!'
		except IOError:
			print "Weights file wasn't found. Generating new connections"
			kern2 = gkern2((3,3),3)
			kern2 = np.identity(3)
			Wr = kernel2connection(kern2, i_shape,o_shape)

			#Wr /= np.sum(Wi,1)
			np.save(r_file,Wr)
		print "Shaaaape!!!!! " + str(Wr.shape)
		if np.sum(Wi,1)[0] != 1:
			Wi /= np.sum(Wi,1).reshape((Wi.shape[0],1))*20
		if np.sum(Wr,1)[0] != 1:
			Wr /= np.sum(Wr,1).reshape((Wr.shape[0],1))
		print np.sum(Wi,0)
		print np.sum(Wi,1)
		#plt.plot(Wi[1,:])
		#plt.show()


		self.Wi= theano.shared( 
				sp.csc_matrix(
				np.asarray( 
				Wi, 
				dtype=input.dtype) ), name ='Wi')
		self.Wr = theano.shared( 
				sp.csc_matrix(
				np.asarray( 
				Wr, 
				dtype=input.dtype) ), name ='Wr')
		# Output of the layer is the sigmoid of the convolved network
		self.state = theano.shared( 
			sp.csc_matrix(
			np.asarray( 
			np.zeros((o_shape[0]*o_shape[1],1)), 
			dtype=input.dtype) ), name ='St')

		self.input = input

		# I could do the same with biases if needed
		#print self.input.get_value().shape
		#print self.Wi.get_value().shape
		self.output = theano.shared( 
			sp.csc_matrix(
			np.asarray( 
			np.zeros((o_shape[0]*o_shape[1],1)), 
			dtype=input.dtype) ), name ='Out')
		#sparse.structured_sigmoid(sparse.structured_dot(self.input, self.Wi))  #T.dot(self.input, self.Wi))
		# input = external + recursive (from layer)
		# self.input = T.dot(input, self.Wi) #+ T.sum(T.dot(self.state,self.Wr),1)

		# out: nx1
		# Wi: mxn
		# outT x WiT : 1xm
		self.yw = sparse.structured_dot(
						sparse.transpose(self.output),
						sparse.transpose(self.Wi))
		# in: nx1
		self.x_yw = sparse.sub(
						sparse.transpose(self.input),
						self.yw)


		# optional: self.output = T.nnet.sigmoid(conv_out+self.output)
		self.params = [self.Wi, self.Wr, self.state, self.output]
		#self.input=input




	# Create the function that solves the previous operations

x = T.matrix('x',dtype = 'float32') #T.matrix('x')   # the data is presented as rasterized images
#y = T.matrix('y')
#x.tag.test_value = np.random.rand(50,50)
#x.flatten()

#weights1 = gkern2(filter_shape[2],sigma).reshape(filter_shape)


input_shape = (10, 2)
inp_filter_shape = (3,3)
inp_filter_sigma = 2
L0_shape = (10,1)
filter_shape = (2,1)
L1_shape = (15,15)

layer0_input = sparse.csc_from_dense(x)

#final_shape = (s2*s3)

sigma = 1

#i_file = 'Wi_' + str(input_shape) + 'x' + str(L0_shape) + '_' + str(filter_shape[0]) + 's' + str(sigma) + '.npy'
#r_file = 'test_Wr.npy'
LR = np.cast['float32'](0.000001)

global generate
generate = False


i_file = 'inh_i_' + str(input_shape) + 'x' + str(input_shape) + '_' + str(filter_shape[0]) + 's' + str(sigma) + '.npy'
r_file = 'inh_r_' + str(input_shape) + 'x' + str(input_shape) + '_' + str(filter_shape[0]) + 's' + str(sigma) + '.npy'
input_layer = HebbianInhibitoryLayer(layer0_input,inp_filter_shape,inp_filter_sigma,input_shape,input_shape, i_file, r_file)

i_file = 'Wi_' + str(input_shape) + 'x' + str(L0_shape) + '_' + str(filter_shape[0]) + 's' + str(sigma) + '.npy'
r_file = 'test_Wr.npy'
layer0 = HebbianAdaptiveLayer(input_layer.output,filter_shape,sigma,input_shape, L0_shape, i_file, r_file)

#layer1 = HebbianAdaptiveLayer(layer0.output,filter_shape,sigma,s1,s2,s2,final_shape,Wi=Wi,Wr=Wr)

layers = [layer0]#, layer1]
out = [layer0.output]#, layer1.output]

inp = T.matrix()
#propagate = theano.function([out],y)

#updates = [(param_i + LR*layer0.state*(x-y)) for param_i in zip(params)]

#update = [(param_i, param_i + LR)]
index=T.lscalar()

csc_mat = sparse.csc_matrix('cscMat', dtype='float32')
qq,ww,ee,rr = sparse.csm_properties(csc_mat)
csc_trans = sparse.CSR(qq,ww,ee,rr)
#trans = theano.function([csc_mat],csc_trans)


Wis = []
Wrs = []
states = []
outs=[]
a = sp.csc_matrix(np.asarray([[0, 1, 1], [0, 0, 0], [1, 0, 0]],dtype='float32'))
print sparse.transpose(a).toarray()

old_W = sparse.csc_matrix('old_W',dtype='float32') # Old weight matrix
pop_i = sparse.csc_matrix('pop_i',dtype='float32') # Input layer
pop_j = sparse.csc_matrix('pop_j',dtype='float32') # Output layer
alpha = T.scalar('alpha',dtype='float32')
'''
new_W = sparse.add(old_W,
					sparse.sub(
						alpha*sparse.structured_dot(sparse.transpose(pop_j), pop_i), 
						sparse.structured_dot(
							sparse.structured_dot(
								sparse.transpose(pop_j), 
								pop_j),
							old_W)
						)
					)

new_W = old_W
#hebbianL = theano.function([old_W], new_W)
print layer0.input
print layer0.output
print layer0.Wi
print old_W
#wi=hebbianL([layer.params[0]])
'''
for layer in layers:
	
	Wis.append( (layer.params[0],  
		#layer.params[0]))
		sparse.add( 
			layer.params[0] , 
			LR*sparse.transpose(
				sparse.structured_dot(layer.output, layer.x_yw)
				)
			)
		))
	
		
		# (layer.params[0], 
		# sparse.add( 
		# 	layer.params[0] , 
		# 	LR*sparse.sub(
		# 		sparse.structured_dot(sparse.transpose(layer.output), layer.input) , 
		# 		sparse.structured_dot(
		# 			sparse.structured_dot(
		# 				sparse.transpose(layer.output), 
		# 				layer.output),
		# 			layer.params[0])
		# 		) 
		# 	)  
		# ))
	
	Wrs.append( (layer.params[1],layer.params[1]) )
	states.append( (layer.params[2], layer.output) )
	outs.append( (layer.params[3], sparse.transpose(
											sparse.structured_sigmoid(
												sparse.structured_dot(
													sparse.transpose(layer.input), 
													layer.params[0]
													)
												)
											)
										) )


#Update inhibitory
Wis.append( (input_layer.params[0], input_layer.params[0]))
Wrs.append( (input_layer.params[1],input_layer.params[1]) )
states.append( (input_layer.params[2], input_layer.output) )
outs.append( (input_layer.params[3], sparse.transpose(
											sparse.structured_sigmoid(
												sparse.add(
													sparse.structured_dot(
														sparse.transpose(input_layer.input), 
														input_layer.params[1]
														),
													sparse.structured_dot(
														sparse.transpose(input_layer.state), 
														input_layer.params[0]
														)
													)

												)
											)
										) )
updates = Wis + Wrs + states + outs


propagate = theano.function(
    [x],
    out,
    updates=updates,
    allow_input_downcast=True,
    mode='ProfileMode'
)

# http://www.deeplearning.net/tutorial/lenet.html#lenet

z=np.random.rand(input_shape[0],input_shape[1]).astype('float32')
#print z.shape
z=z.flatten().reshape((input_shape[0]*input_shape[1],1)).transpose()
#print z.shape
import time as t
plt.ion()
#fig = plt.figure()
fig, ax = plt.subplots(1,4)
p = ax[0].imshow( z.reshape(input_shape) ,vmin = 0, vmax = 1, interpolation='none',aspect='auto')
#plt = fig.add_subplot(122)
w = ax[2].imshow( np.zeros( (input_shape[0]*input_shape[1], L0_shape[0]*L0_shape[1]) ) ,vmin=0, interpolation='none')

out_plot = ax[3].imshow( np.zeros( (L0_shape[0]*L0_shape[1], L0_shape[0]*L0_shape[1]) ) ,vmin=0, vmax=1, interpolation='none')

orig_in = ax[1].imshow( z.reshape(input_shape) ,vmin = 0, vmax = 1,aspect='auto',interpolation='none')


#w = ax[1].imshow( z.reshape((s1,s2)) )
#plt.show()
#n_inputs = s1
CF = 2
freqs = np.arange(7)/4.*input_shape[0]
freqs = [2,4,6,8]
logs = np.arange((input_shape[1]+2))/float(input_shape[1]+2)
logs = logs[1:input_shape[1]+1]
logs = [0.1,0.9]
print logs
noise = 0.0
n=1
global audio_input, input_means, input_std, input_alpha, input_out
audio_input = 0
input_alpha = 1

input_std = np.ones(input_shape)
input_means = np.zeros(input_shape)
input_out = np.zeros(input_shape)


def sigmoid(x,k):
  return 1 / (1 + np.exp(-x))
def something(x,logs):
	#print x.shape
	#print logs.shape
	aux = (np.log(x)-np.log(logs))#/np.log((1-logs))
	#print aux
	#print logs
	#print aux
	return aux
def finiteIsMean(x,mu):
	x[np.isfinite(x) == False] = mu[np.isfinite(x) == False]
	return x
def average_input(new_input):
	global input_means, input_std, input_alpha, input_out
	# new_input[np.isfinite(new_input) == False] = input_means[np.isfinite(new_input) == False]
	#print np.isfinite(new_input)
	new_input = np.apply_along_axis(finiteIsMean,0,new_input, input_means)
	#print np.isfinite(new_input)
	means = input_means*(1-input_alpha)+input_alpha*new_input
	std = np.sqrt(np.power(input_std,2)*(1-input_alpha)+np.power(new_input-input_means,2)*input_alpha)

	out = (new_input-input_means)/input_std # 0.5 + 0.5*np.tanh((new_input-means)/std) #np.sqrt(2))
	out = input_means
	input_means = means
	input_std = std
	#out[np.isfinite(out) == False] = input_out[np.isfinite(out) == False]
	out = np.apply_along_axis(finiteIsMean,0,out, input_out)
	input_out = out
	#print 'max: ' + str(np.max(out))
	#print 'min: ' + str(np.min(out))
	return out
	
def dataIsAudio(x,logs,shap):
	
	#print x.shape
	z = np.apply_along_axis(something,1,x.reshape(shap[0],1,order='C'),logs)
	#print z.shape
	#aux = np.maximum(z+2,0)
	z=average_input(z)
	#print aux
	#z = sigmoid(np.maximum(z+2,0))
	#print z.shape
	#print np.min(z)
	z=z.reshape((shap[0]*shap[1],1),order='C')
	return z
print logs


for n_epoch in range(1000):
#def updat(event):
	#global n, CF, n_inputs,freqs,noise
	#z=np.random.rand(s1,s2)
	#p.set_data(z)
	#fig.canvas.draw()
	fft_input,CF,SNR=generateRandomSignalBut(n_epoch,CF,size=input_shape[0],f=freqs, but=0,noi=noise)
	orig_in.set_data(input_layer.output.get_value().toarray().reshape(input_shape,order='C'))
	#logs = np.array([ 0.1,  0.2,  0.3,  0.4,  0.6])
	#audio_input = average_input(fft_input, audio_input, alpha)
	z=dataIsAudio(fft_input, logs, input_shape)
	now = t.time()
	outp = propagate(z)
	print t.time()-now
	p.set_data( z.reshape(input_shape,order='C') )
	#print z.shape
	#w.set_data( outp.toarray().reshape((s1,s2)) )
	w.set_data(layer0.Wi.get_value().toarray())
	out_plot.set_data(outp[0].toarray().reshape(L0_shape))
	#w.set_data(Wi)
	fig.canvas.draw()
	if n_epoch >0:
		asd = input('now')


#fig.canvas.mpl_connect('button_press_event', updat)
#plt.show()
