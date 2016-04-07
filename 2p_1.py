import numpy as np
import matplotlib.pyplot as plt

def generateRandomSignalBut(epoch,center_F,size=10,f=[1, 2,3,4, 5,6,7,8,9],but=0,noi=0.2, n_harmonics=0):
    if epoch%3 == 0:
    	x= np.random.rand(size)*noi
    else: 
    	try: 
    		x=x
    	except UnboundLocalError:
    		x=np.zeros(size)
    Nx=0
    if epoch%50 == 0:
        it=(epoch%(int(10000/30)))%len(f)
        center_F = np.random.choice(f,1)/2
        #print "New CF: "  + str(center_F) + " from " + str(f)
        while center_F==but:
            center_F = np.random.choice(f,1)/2
    for i in range(1):
    	#print max(0,int(center_F)+i-4)
        x[max(0,int(center_F)+i)]=0.9
        Nx+=0.9
    SNR=pow(Nx,2)/pow(np.sum(x)-Nx,2)
    x[x<0.001]=0
    
    return x,center_F,SNR

def update_weights_oja(w,LR,i,x):
	w = w + LR*(inp*x - x*x*w)
	return w

def update_weights_epfl(w,LR,i,x):
	global t_p, t_n, We, Wmax, Wmin
	A=(Wmax-w)/t_p
	A_=(w-Wmin)/t_n
	auxLR = LR*(A - A_)
	w = w - auxLR*inp*x + LR*A*We*w
	#print w
	return w

def update_weights_Vogels2011(w,LR,i,x):
	w = w + LR*(inp*x - x*x*w)
	return w

#Init experiments
S=10
aux = np.array(range(10),dtype='float32')
mu = 5.
sigma = 2.
LR_all=0.00001
LR_US=0.01
global t_p, t_n, We, Wmax, Wmin
t_p = 1./0.1 # Normally 1/tau_n * 1/tau_p <0 and >-(w//1-w)), with w~stability weight 
t_n = 1./0.9
We = -0.7
Wmax=0.9
Wmin=0.0

w0 = (1/(sigma*np.sqrt(2*np.pi)))*np.exp(-((aux-5)**2)/(2*sigma**2))

n_experiments = 20
W=np.zeros((n_experiments,S))
for exp in range(n_experiments):


	# initialize weights
	
	w = w0

	#plt.plot(w)
	#plt.show()
	#Initialize LR
	
	LR=LR_all

	CF=0

	for i in range(2000):
		# generate input
		inp, CF, snr = generateRandomSignalBut(i,CF, size=S)
		# Normalize input
		# compute output
		x = np.tanh(np.sum(inp*w))
		# Change LR for US
		if CF == 4:
			LR = LR_US
		else:
			LR=LR_all
		# update weights (Oja)
		#w=update_weights_oja(w,LR,inp,x)
		w=update_weights_epfl(w,LR,inp,x)
	W[exp,:]=w


#plt.boxplot(W)
plt.plot(np.mean(W,0),'b')
plt.plot(w0,'r')
plt.savefig('2phase.png')
plt.show()