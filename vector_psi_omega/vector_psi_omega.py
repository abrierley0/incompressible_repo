
# 2D Lid-driven Cavity
# VECTOR-POTENTIAL VORTICITY FORMULATION
# Written by Mr A. J. Brierley
# Cranfield University, Bedfordshire, UK
# 03/06/2025


import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import cm
plt.rcParams['animation.html'] = 'html5'

nx = 7
ny = 7
nz = 7
lx = 1.0
ly = 1.0
lz = 1.0
dx = lx/(nx-1)
dy = ly/(ny-1)
dz = lz/(nz-1)

Ut = 3.2 # top wall velocity

# Specify initial values for the streamfunction (psi) and vorticity (Omega)
# at t = 0 on the whole domain
# Then specify conditions that will produce values at the boundaries at t = 0
# And be enforced as the solution is marched through time
psix = np.zeros([nx,ny,nz])
psiy = np.zeros([nx,ny,nz])
psiz = np.zeros([nx,ny,nz])

u = np.zeros([nx,ny,nz])
v = np.zeros([nx,ny,nz])
w = np.zeros([nx,ny,nz])

#omega = 10*np.random.rand(7,7,7)
omegax = np.zeros([nx,ny,nz])
omegay = np.zeros([nx,ny,nz])
omegaz = np.zeros([nx,ny,nz])

#print(omega)

#------------------------------------------------------
# VECTOR-POTENTIAL BOUNDARY CONDITIONS
#------------------------------------------------------
# Left wall
for j in range(ny):     # Not inclusive
    for k in range(nz):
        psix[0,j,k] = psix[1,j,k]
        psiy[0,j,k] = 0.0
        psiz[0,j,k] = 0.0

# Bottom wall
for i in range(nx):
    for z in range(nz):
        psix[i,0,k] = 0.0
        psiy[i,0,k] = psiy[i,0,k]
        psiz[i,0,k] = 0.0

# Right wall 
for j in range(ny):
    for z in range(nz):
        psix[nx-1,j,k] = psix[nx-2,j,k]  # Here, nx will try to index 13 which is outside the 0 to 13 exclusive range
        psiy[nx-1,j,k] = 0.0
        psiz[nx-1,j,k] = 0.0

# Top wall
for i in range(nx):
    for z in range(nz):
        psix[i,ny-1,k] = Ut
        psiy[i,ny-1,k] = psiy[i,ny-2,k]
        psiz[i,ny-1,k] = 0.0

# Front wall
for i in range(nx):
    for j in range(ny):
        psix[i,j,0] = 0.0
        psiy[i,j,0] = 0.0
        psiz[i,j,0] = psiz[i,j,1]

# Back wall
for i in range(nx):
    for j in range(ny):
        psix[i,j,nz-1] = 0.0
        psiy[i,j,nz-1] = 0.0
        psiz[i,j,nz-1] = psiz[i,j,nz-2]

#------------------------------------------------
# VORTICITY BOUNDARY CONDITIONS 
#------------------------------------------------
# Left wall
for j in range(ny):
    for k in range(nz):
        omegax[0,j,k] = 0.0
        omegay[0,j,k] = - w[1,j,k]/dx
        omegaz[0,j,k] = v[1,j,k]/dx
# Right wall
for j in range(ny):
    for k in range(nz):
        omegax[nx-1,j,k] = 0.0
        omegay[nx-1,j,k] = w[nx-2,j,k]/dx
        omegaz[nx-1,j,k] = -v[nx-2,j,k]/dx
# Lower wall 
for i in range(nx):  # starting from 0 and exclusive
    for k in range(nz):
        omegax[i,0,k] = w[i,1,k]/dy
        omegay[i,0,k] = 0.0
        omegaz[i,0,k] = -u[i,1,k]/dy
# Front wall
for i in range(nx):
    for j in range(ny):
        omegax[i,j,0] = -v[i,j,1]/dz
        omegay[i,j,0] = u[i,j,1]/dz
        omegaz[i,j,0] = 0.0
# Back wall 
for i in range(nx):
    for j in range(ny):
        omegax[i,j,nz-1] = v[i,j,nz-2]/dz
        omegay[i,j,nz-1] = -u[i,j,nz-2]/dz
        omegaz[i,j,nz-1] = 0.0
# Top wall 
for i in range(nx):
    for k in range(nz-1):  # NOTE: dodgy bit
        omegax[i,ny-1,k] = -w[i,ny-2,k]/dy
        omegay[i,ny-1,k] = 0.0
        omegaz[i,ny-1,k] = -u[i,ny-1,k]/dy

# Solution Storage
psixsol = []
psiysol = []
psizsol = []
psixsol.append
psiysol.append
psizsol.append
omegasol = []
omegasol.append(omega)

# Time stepping
t = 0.0  # initial time and time counter
v = 0.05  # nu
dt = min(0.25*dx*dx/v, 4*v/Ut/Ut)
tend = 1000*dt
print('dt =', dt, 's')
print('Re =', Ut*lx/v)

#-----------------------------------------------------
# SOLVE THREE VECTOR-POTENTIAL POISSON EQUATIONS
#-----------------------------------------------------
maxIt = 100
err = 1e5
tol = 1e-3
beta = 1.5
# Start main time loop
while t < tend:
    it = 0
    while err > tol and it < maxIt:
        psikx = psix.copy() # Stores the previous value
        psiky = psiy.copy()
        psikz = psiz.copy()
        # Solve vector-potential Poisson for psix
        for i in range(1,nx-1):
            for j in range(1,ny-1):
                for k in range(1,nz-1):
                    psix[i,j,k] = dx**2*dy**2*dz**2*omegax[i,j,k] + \
                                    dy**2*dz**2*(psix[i+1,j,k] + psix[i-1,j,k]) + \
                                    dx**2*dz**2*(psix[i,j+1,k] + psix[i,j-1,k]) + \
                                    dx**2*dy**2*(psix[i,j,k+1] + psix[i,j,k-1])
                    psix[i,j,k] = beta * psix[i,j,k]/(2*(dz**2*dy**2 + dx**2*dz**2 + dx**2*dy**2)) + (1-beta)*psikx[i,j,k]

        # Solve vector-potential Poisson for psiy
        for i in range(1,nx-1):
            for j in range(1,ny-1):
                for k in range(1,nz-1):
                    psiy[i,j,k] = dx**2*dy**2*dz**2*omegay[i,j,k] + \
                                    dy**2*dz**2*(psiy[i+1,j,k] + psiy[i-1,j,k]) + \
                                    dx**2*dz**2*(psiy[i,j+1,k] + psiy[i,j-1,k]) + \
                                    dx**2*dy**2*(psiy[i,j,k+1] + psiy[i,j,k-1])
                    psiy[i,j,k] = beta * psiy[i,j,k]/(2*(dz**2*dy**2 + dx**2*dz**2 + dx**2*dy**2)) + (1-beta)*psiky[i,j,k]

        # Solve vector-potential Poisson for psiz
        for i in range(1,nx-1):
            for j in range(1,ny-1):
                for k in range(1,nz-1):
                    psiz[i,j,k] = dx**2*dy**2*dz**2*omegaz[i,j,k] + \
                                    dy**2*dz**2*(psiz[i+1,j,k] + psiz[i-1,j,k]) + \
                                    dx**2*dz**2*(psiz[i,j+1,k] + psiz[i,j-1,k]) + \
                                    dx**2*dy**2*(psiz[i,j,k+1] + psiz[i,j,k-1])
                    psiz[i,j,k] = beta * psiz[i,j,k]/(2*(dz**2*dy**2 + dx**2*dz**2 + dx**2*dy**2)) + (1-beta)*psikz[i,j,k]

        err_x = np.linalg.norm(psix.ravel() - psikx.ravel())
        err_y = np.linalg.norm(psiy.ravel() - psiky.ravel())
        err_z = np.linalg.norm(psiz.ravel() - psikz.ravel())
        err = max(err_x, err_y, err_z)
        print(err)
        it = it + 1

    t = t + dt

print(f"psix = ", psix)
print(f"psiy = ", psiy)
print(f"psiz = ", psiz)
print(f"iteration =", it)
print(f"err =", err)



