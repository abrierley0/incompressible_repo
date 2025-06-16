import numpy as np
from poisson import poisson

nx = 13
ny = 13
nz = 13
L = 1.0
H = 1.0
W = 1.0
dx = L/(nx-1)
dy = H/(ny-1)
dz = W/(nz-1)

psi = np.zeros((nz,nx,ny))
omega = np.zeros((nz,nx,ny))
velocity = np.zeros((nz,nx,ny))

t = 1e-5
tend = 10
tol = 1e-5
beta = 1.5

# Solve vector-potential Poisson equations for the vorticity
while t < tend:
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                rhs = dx**2*dy**2*dz**2*omega[i,j,k] + dy**2*dz**2*(psi[i+1,j,k] + psi[i-1,j,k]) + dx**2*dz**2*(psi[i,j+1,k] + psi[i,j-1,k]) + dx**2*dy**2*(psi[i,j,k+1]+psi[i,j,k-1])
                rhs *= beta/(2*dz**2*dy**2 + dx**2*dz**2 + dx**2*dy**2)

