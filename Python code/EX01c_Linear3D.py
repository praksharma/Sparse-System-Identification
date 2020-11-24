#import os  # clear screen
import numpy as np   # matrix calc
from scipy.integrate import odeint # scientific computation (ode solver)
from utils.methods import poolData, sparsifyDynamics, sparseGalerkin , sparseGalerkin3D
import matplotlib.pyplot as py

#os.system('clear')

# Generate data
A=np.array([[-0.1, 2, 0],[ -2, -0.1, 0],[0, 0, -0.3]])
polyorder=2     # search space up to fifth order polynomials
usesine=0       # no trig functions
n=3             # 2D system
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
    
    return [rhs[0],rhs[1],rhs[2]]

tspan=np.arange(0,50.01,0.01,dtype=float)  # time span
#tspan1=np.transpose(tspan)
x0= np.array([2,0,1])                 # initial conditions

#temp=1e-10*np.ones((1,n),dtype=np.int32)
#time,pos=odeint(lambda t,x: rhs(x), x0, tspan, atol=1e-10, rtol=1e-10)
x=odeint(dUdt, x0, tspan, atol=1e-10, rtol=1e-10)
print('ODEint successfully ran for exact solution')

#Compute derivative
eps=0.01         # Noise strength
dx=np.zeros((len(x),3))
for i in range(1,len(x)):
    dx[i,:]=np.matmul(A,(x[i,:]))

dx=dx+ eps*np.random.standard_normal(np.shape(dx)) # Adding noise (normal dist)

# Pool data (building library of non-linear time series)
theta=poolData(x,n,polyorder,usesine)
print('Non-linear time series build successful')
m=np.shape(theta)[1]  # zero is row, 1 is column
    

# Compute sparse regression: sequential least squares
Lambda=0.085     # lambda is the sparsification knob
Xi=sparsifyDynamics(theta,dx,Lambda,n)
print('Building SIND model successful')

# integrate true and identified systems
xA=odeint(dUdt, x0, tspan, atol=1e-10, rtol=1e-10) # Exact solution
xB=odeint(lambda x,t: sparseGalerkin3D(x,t,Xi,polyorder,usesine),x0,tspan,atol=1e-10, rtol=1e-10) # SIND solution
print('ODE int successfully ran for SIND model')

# Plotting solutions
fig = py.figure(1)
ax = fig.gca(projection='3d')
ax.plot(xA[:,0],xA[:,1],xA[:,2],"lawngreen")   # disp-X vs disp-y for exact solution
ax.plot(xB[:,0],xB[:,1],xB[:,2],'--k')    # disp-X vs disp-y for SIND solution
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('$x_3$')
ax.view_init(25,45)     # Camera position
py.legend(['Exact solution','SIND solution'])
figure = py.gcf()
figure.set_size_inches(12, 9)
py.savefig("Linear 3D figures/Linear3D.jpg",dpi=500)

py.figure(2)
py.plot(tspan,xA[:,0],'r')
py.plot(tspan,xA[:,1],'b')
py.plot(tspan,xA[:,2],"lawngreen")
py.plot(tspan,xB[:,0],'--k')
py.plot(tspan,xB[:,1],'--k')
py.plot(tspan,xB[:,2],'--k')
py.xlabel("Time-steps")
py.ylabel('u')
py.legend(['True $x_1$','True $x_2$','True $x_3$','SIND solution'])
figure = py.gcf()
figure.set_size_inches(12, 9)
py.savefig("Linear 3D figures/Linear3D-timeseries.jpg",dpi=500)


#Plotting errors
residual1=abs(xA[:,0]-xB[:,0])
residual2=abs(xA[:,1]-xB[:,1])
residual3=abs(xA[:,2]-xB[:,2])

py.figure(3)
py.plot(tspan,residual1)
py.plot(tspan,residual2)
py.plot(tspan,residual3)
py.legend(['Initial condition 1','Initial condition 2','Initial condition 3'])
py.xlabel("Time-steps")
py.ylabel('abs(Residual)')
figure = py.gcf()
figure.set_size_inches(12, 9)
py.savefig('Linear 3D figures/residual-plot.jpg',dpi=500)

print(Xi)
np.savetxt('Linear 3D figures/SIND equation.txt',Xi,fmt="%4.4f")