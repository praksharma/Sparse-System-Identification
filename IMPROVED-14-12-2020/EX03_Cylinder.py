#import os  # clear screen
import numpy as np   # matrix calc
from scipy.integrate import odeint # scientific computation (ode solver)
from utils.methods import poolData, sparsifyDynamics, sparseGalerkin 
import matplotlib.pyplot as py
#os.system('clear')

# Importing POD coefficients
from scipy.io import loadmat
POD_coefficients = loadmat('.\Data\POD Coefficients\PODcoefficients.mat')
alpha=POD_coefficients['alpha']
alphaS=POD_coefficients['alphaS']

# Generate data
dt=0.02
r=2

# Load data from first run and compute derivative
temp1=alpha[1-1:5000-1,1-1:r]
temp2=np.reshape(alphaS[1-1:5000-1,1-1],(4999,1))
x=np.concatenate([temp1,temp2],axis=1)
M = np.shape(x)[1]

A=np.array([[-0.1, 2],[ -2, -0.1]])
polyorder=5     # search space up to fifth order polynomials
usesine=0       # no trig functions
n=2             # 2D system

def dUdt(x1,t1):
    rhs=np.dot(A,np.power(x1,3))
    return [rhs[0],rhs[1]]

tspan=np.arange(0,25.01,0.01,dtype=float)  # time span
#tspan1=np.transpose(tspan)
x0= np.array([2,0])                 # initial conditions

#temp=1e-10*np.ones((1,n),dtype=np.int32)
#time,pos=odeint(lambda t,x: rhs(x), x0, tspan, atol=1e-10, rtol=1e-10)
x=odeint(dUdt, x0, tspan, atol=1e-10, rtol=1e-10)
print('ODEint successfully ran for exact solution')

#Compute derivative
eps=0.05         # Noise strength
dx=np.zeros((len(x),2))
for i in range(1,len(x)):
    dx[i,:]=np.matmul(A,np.power((x[i,:]),3))

dx=dx+ eps*np.random.standard_normal(np.shape(dx)) # Adding noise (normal dist)

# Pool data (building library of non-linear time series)
theta=poolData(x,n,polyorder,usesine)
print('Non-linear time series build successful')
m=np.shape(theta)[1]  # zero is row, 1 is column
    

# Compute sparse regression: sequential least squares
Lambda=0.05     # lambda is the sparsification knob
Xi=sparsifyDynamics(theta,dx,Lambda,n)
print('Building SIND model successful')

# integrate true and identified systems
xA=odeint(dUdt, x0, tspan, atol=1e-10, rtol=1e-10) # Exact solution
xB=odeint(lambda x,t: sparseGalerkin(x,t,Xi,polyorder,usesine),x0,tspan,atol=1e-10, rtol=1e-10) # SIND solution
print('ODE int successfully ran for SIND model')

# Plotting solutions
py.figure(1)
figure = py.gcf()
figure.set_size_inches(12, 9)
py.plot(xA[:,0],xA[:,1])    # disp-X vs disp-y for exact solution
py.plot(xB[:,0],xB[:,1])    # disp-X vs disp-y for SIND solution
py.xlabel('x_1')
py.ylabel('x_2')
py.legend(['Exact','SIND'])
py.savefig('Cylinder figures/x vs y plot.jpg',dpi=500)


py.figure(2)
figure = py.gcf()
figure.set_size_inches(12, 9)
py.plot(tspan,xA[:,0],'-r')
py.plot(tspan,xA[:,1],'-b')    # exact solution plot
py.plot(tspan,xB[:,0],'--k')
py.plot(tspan,xB[:,1],'--k')      # SIND solution
py.xlabel("Time-steps")
py.ylabel('u')
py.legend(['True x_1','True x_2','SIND solution'])
py.savefig("Cylinder figures/Cubic2D-u vs time.jpg",dpi=500)

#Plotting errors
residual1=abs(xA[:,0]-xB[:,0])
residual2=abs(xA[:,1]-xB[:,1])

py.figure(3)
figure = py.gcf()
figure.set_size_inches(12, 9)
py.plot(tspan,residual1)
py.plot(tspan,residual2)
py.legend(['Initial condition 1','Initial condition 2'])
py.xlabel("Time-steps")
py.ylabel('abs(Residual)')
#py.suptitle('ln of absolute residual')
py.savefig('Cylinder figures/residual-plot.jpg',dpi=500)
 
print(Xi)
np.savetxt('Cylinder figures/SIND equation.txt',Xi,fmt="%4.4f")