#import os  # clear screen
import numpy as np   # matrix calc
from scipy.integrate import odeint # scientific computation (ode solver)
from utils.methods import poolData, sparsifyDynamics, sparseGalerkin 
import matplotlib.pyplot as py
#os.system('clear')

# Generate data
A=np.array([[-0.1, 2],[ -2, -0.1]])
polyorder=5     # search space up to fifth order polynomials
usesine=0       # no trig functions
n=2             # 2D system
#A=np.array([[-0.1, 2],[ -2, -0.1]])     # dynamics
#rhs=lambda x: A*x                       # ODE right hand side

def dUdt(x1,t1):
    
    
    # I have to write the equations manually
    
    # Method 1
    
    #rhs1=A[0,0]*x[0]+A[0,1]*x[1] 
    #rhs2=A[1,0]*x[0]+A[1,1]*x[1] 
    #return [rhs1,rhs2]
    
    # Method 2 (a short one basically)
    
    rhs=np.dot(A,x1)
    
    return [rhs[0],rhs[1]]

tspan=np.arange(0,55.01,0.01,dtype=float)  # time span
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
    dx[i,:]=np.matmul(A,(x[i,:]))

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
py.plot(tspan,xA[:,0],'r')
py.plot(tspan,xB[:,0],'--k')
py.xlabel("Time-steps")
py.ylabel('u')
py.legend(['Exact solution','SIND solution'])
py.savefig("Linear 2D figures/Linear2D-column1.jpg",dpi=500)




py.figure(2)
figure = py.gcf()
figure.set_size_inches(12, 9)
py.plot(tspan,xA[:,1],'r')
py.plot(tspan,xB[:,1],'--k')
py.xlabel("Time-steps")
py.ylabel('u')
py.legend(['Exact solution','SIND solution'])
py.savefig("Linear 2D figures/Linear2D-column2.jpg",dpi=500)


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
py.savefig('Linear 2D figures/residual-plot.jpg',dpi=500)

print(Xi)
np.savetxt('Linear 2D figures/SIND equation.txt',Xi,fmt="%4.4f")