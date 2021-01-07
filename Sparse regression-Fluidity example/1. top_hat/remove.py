import numpy as np 
x=np.transpose(np.loadtxt('DataFile.txt'))[:,1]
x=np.transpose(np.asmatrix(x))
grad=np.gradient(np.transpose(x[:,0]),axis=1)

tspan=np.transpose(np.loadtxt('DataFile.txt'))[:,0]
dx=np.zeros((len(x),int(np.shape(x)[1])))
i=0
numerator=np.transpose(x[:,i])
dx[:,i]=np.gradient(numerator,axis=1)/np.gradient(tspan)