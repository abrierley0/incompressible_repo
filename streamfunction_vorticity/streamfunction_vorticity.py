import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import cm

nx = 13
ny = 13
lx = 1.0
ly = 1.0
dx = lx/(nx-1)
dy = ly/(ny-1)

Ut = 1.0  # top wall velocity, m/s

psi0 = np.zeros([nx,ny])
w0 = np.zeros([nx,ny]) 
psi_wall = 0.0

# apply boundary conditions for vorticity
w0[:,0] = 2.0*(psi_wall - psi0[:,1])/(dx*dx) # left wall
w0[:,-1] = 2.0*(psi_wall - psi0[:,-2])/(dx*dx) # right wall
w0[-1,:] = 2.0*(psi_wall - psi0[-2,:])/dy/dy - 2.0*Ut/dy # top wall
w0[0,:] = 2.0*(psi_wall - psi0[1,:])/dy/dy # bottom wall

psisol = []
psisol.append(psi0)
omegasol = []
omegasol.append(w0)

# simulation parameters
beta = 1.5
tol = 1e-3

t = 0.0 # initial time and time counter
v = 0.05
dt = min(0.25*dx*dx/v, 4*v*Ut*Ut)
tend = 1000*dt
print('dt =', dt, 's')
print('Re = ', Ut*lx/v)

while t < tend:

    # start by solving Poisson equation for the streamfunction
        it = 0
        err = 1e5
        # present omega is wn
        wn = wsol[-1] # copy to omega from the previous omega
        psi = psi[-1].copy() # copy to psi from the previous psi
        while err > tol and it < maxIt:
            psik = np.zeros_like(psi) # return array of zeros with same shape as size as psi
            psik[1:-1, 1:-1] = psi[1:-1,1:-1]
            # loop over interior points
            for i in range(1,nx-1):
                for j in range(1,ny-1):
                    rhs = (dx*dy)**2*wn[j,i] + dx**2*(psi[j,i+1]+psi[j,i-1]) + dx**2*(psi[j+1,i]+psi[j,i-1])
                    rhs *= beta/2.0/(dx**2 + dy**2)
                    psi(i,j) = rhs + (1.0 - beta)*psi(j,i)
            err = np.linalg.norm(psi.ravel() - psik.ravel())
            it += 1
        psisol.append(psi)

    # Next solve the discretised 2D vorticity transport equation
    w = np.zeros_like(wn)

    # x convection term, discretised
    Cx = -(psi[2:,1:-1] - psi[:-2,1:-1])/2.0/dy * (wn[1:-1,2:] - wn[1:-1:-2])/2.0/dx
    # y convection term, discretised
    Cy = (wn[2:,1:-1] - wn[:-2,1:-1])/2.0/dy * (psi[1:-1,2:] - psi[1:-1:-2])/2.0/dx
    # diffusion in x, discretised
    Dx = (wn[1:-1,2:] - 2.0*wn[1:-1,1:-1] + wn[1:-1,:-2])/dx/dx
    # diffusion in y, discretised
    Dy = (wn[2:,1:-1] - 2.0*wn[1:-1,1:-1] + wn[:-2,1:-1])/dy/dy
    rhs = Cx + Cy + v*(Dx + Dy)
    w[1:-1,1:-1] = wn[1:-1,1:-1] + dt * rhs

    # apply boundary conditions on vorticity
    w[:,0] = 2.0*(psi_wall - psi[:,1])/(dx*dx) # left wall
    w[:,-1] = 2.0*(psi_wall - psi[:,-2])/(dx*dx) # right wall
    w[-1,:] = 2.0*(psi_wall - psi[-2,:])/dy/dy - 2.0*Ut/dy # top wall
    w[0,:] = 2.0*(psi_wall - psi[1,:])/dy/dy # bottom wall

    omegasol.append(w)

    t += dt














