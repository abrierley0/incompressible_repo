import numpy as np
import matplotlib.pyplot as plt

# VECTOR-POTENTIAL VORTICITY FORMULATION
#
# Written by Mr A. J. Brierley
#
# Centre for Computational Engineering Sciences (CES)
# Cranfield University
# Bedfordshire
# UK
#
# adam.brierley@cranfield.ac.uk
#
# 2nd July 2025

nx = 7
ny = 7
nz = 7
Lx = 1.0
Ly = 1.0
Lz = 1.0
dx = Lx/(nx-1)
dy = Ly/(ny-1)
dz = Lz/(nz-1)


# Physical parameters
Ϟ = 0.05
Ut = 3.2
Re = Ut*Lx/Ϟ
print()
print(f"REYNOLDS' NUMBER IS {Re}")
print()



# Initialise arrays
#ψ0 = np.zeros([nx,ny,nz])
ψx0 = np.zeros([nx,ny,nz])
ψy0 = np.zeros([nx,ny,nz])
ψz0 = np.zeros([nx,ny,nz])

Ωx0 = np.zeros([nx,ny,nz])
Ωy0 = np.zeros([nx,ny,nz])
Ωz0 = np.zeros([nx,ny,nz])

u0 = np.zeros([nx,ny,nz])
v0 = np.zeros([nx,ny,nz])
w0 = np.zeros([nx,ny,nz])


#---------------------------------------------
# VECTOR-POTENTIAL (ψ) BOUNDARY CONDITIONS
#---------------------------------------------
# according to Tokunaga (1992),

for j in range(1,ny-1):
    for k in range(1,nz-1):
        ψx0[0,j,k] = ψx0[1,j,k]         # Left wall
        ψy0[0,j,k] = 0.0
        ψz0[0,j,k] = 0.0  
        ψx0[nx-1,j,k] = ψx0[nx-2,j,k]   # Right wall
        ψy0[nx-1,j,k] = 0.0
        ψz0[nx-1,j,k] = 0.0

for i in range(1,nx-1):
    for j in range(1,ny-1):
        ψx0[i,j,0] = 0.0                # Front wall
        ψy0[i,j,0] = 0.0
        ψz0[i,j,0] = ψz0[i,j,1]       
        ψx0[i,j,nz-1] = 0.0             # Back wall
        ψy0[i,j,nz-1] = 0.0
        ψz0[i,j,nz-1] = ψz0[i,j,nz-2]

for i in range(1,nx-1):
    for k in range(1,ny-1):
        ψx0[i,0,k] = 0.0                # Bottom wall
        ψy0[i,0,k] = ψy0[i,1,k]   
        ψz0[i,0,k] = 0.0
        
        #---------------------
        # TOP WALL BOUNDARY CONDITION
        #---------------------
        # NOTE: Check top wall derivation
        ψx0[i,ny-1,k] = Ut                # Top wall
        ψy0[i,ny-1,k] = ψy0[i,ny-2,k]   
        ψz0[i,ny-1,k] = 0.0


# Vector-potential edge points
for j in range(1,ny-1):
    u[0,j,0] = (u[1,j,0] + u[0,j,1])/2.0                            # Front-left edge
    v[0,j,0] = (v[1,j,0] + v[0,j,1])/2.0
    w[0,j,0] = (w[1,j,0] + w[0,j,1])/2.0
    u[nx-1,j,0] = (u[nx-2,j,0] + u[nx-1,j,1])/2.0                   # Front-right edge
    v[nx-1,j,0] = (v[nx-2,j,0] + v[nx-1,j,1])/2.0
    w[nx-1,j,0] = (w[nx-2,j,0] + w[nx-1,j,1])/2.0
    u[nx-1,j,nz-1] = (u[nx-2,j,nz-1] + u[nx-1,j,nz-2])/2.0          # Back-right edge
    v[nx-1,j,nz-1] = (v[nx-2,j,nz-1] + v[nx-1,j,nz-2])/2.0
    w[nx-1,j,nz-1] = (w[nx-2,j,nz-1] + w[nx-1,j,nz-2])/2.0
    u[0,j,nz-1] = (u[1,j,nz-1] + u[0,j,nz-2])/2.0                   # Back-left edge
    v[0,j,nz-1] = (v[1,j,nz-1] + v[0,j,nz-2])/2.0
    w[0,j,nz-1] = (w[1,j,nz-1] + w[0,j,nz-2])/2.0

for z in range(1,nz-1):
    u[0,0,z] = (u[1,0,z] + u[0,1,z])/2.0                            # Bottom-left edge
    v[0,0,z] = (v[1,0,z] + v[0,1,z])/2.0        
    w[0,0,z] = (w[1,0,z] + w[0,1,z])/2.0
    u[nx-1,0,z] = (u[nx-2,0,z] + u[0,1,z])/2.0                      # Bottom-right edge
    v[nx-1,0,z] = (v[nx-2,0,z] + v[0,1,z])/2.0        
    w[nx-1,0,z] = (w[nx-2,0,z] + w[0,1,z])/2.0
    u[nx-1,ny-1,z] = (u[nx-2,ny-1,z] + u[nx-1,ny-2,z])/2.0          # Top-right edge
    v[nx-1,ny-1,z] = (v[nx-2,ny-1,z] + v[nx-1,ny-2,z])/2.0        
    w[nx-1,ny-1,z] = (w[nx-2,ny-1,z] + w[nx-1,ny-2,z])/2.0
    u[0,ny-1,z] = (u[0,ny-2,z] + u[1,ny-1,z])/2.0                   # Top-left edge
    v[0,ny-1,z] = (v[0,ny-2,z] + v[1,ny-1,z])/2.0        
    w[0,ny-1,z] = (w[0,ny-2,z] + w[1,ny-1,z])/2.0

for i in range(1,nx-1):
    u[i,0,0] = (u[i,1,0] + u[i,0,1])/2.0                            # Front-bottom edge
    v[i,0,0] = (v[i,1,0] + v[i,0,1])/2.0        
    w[i,0,0] = (w[i,1,0] + w[i,0,1])/2.0
    u[i,0,nz-1] = (u[i,1,nz-1] + u[i,0,nz-2])/2.0                   # Back-bottom edge
    v[i,0,nz-1] = (v[i,1,nz-1] + v[i,0,nz-2])/2.0         
    w[i,0,nz-1] = (w[i,1,nz-1] + w[i,0,nz-2])/2.0 
    u[i,ny-1,0] = (u[i,ny-1,1] + u[i,ny-2,0])/2.0                   # Front-top edge
    v[i,ny-1,0] = (v[i,ny-1,1] + v[i,ny-2,0])/2.0         
    w[i,ny-1,0] = (w[i,ny-1,1] + w[i,ny-2,0])/2.0 
    u[i,ny-1,nz-1] = (u[i,ny-2,nz-1] + u[i,ny-1,nz-2])/2.0          # Back-top edge
    v[i,ny-1,nz-1] = (v[0,ny-2,nz-1] + v[1,ny-1,nz-2])/2.0        
    w[i,ny-1,nz-1] = (w[0,ny-2,nz-1] + w[1,ny-1,nz-2])/2.0

# Vector-potential corner points
ψx0[0,0,0] = (ψx0[1,0,0] + ψx0[0,1,0] + ψx0[0,0,1]) / 3.0                                           # Lower bottom left 
ψy0[0,0,0] = (ψy0[1,0,0] + ψy0[0,1,0] + ψy0[0,0,1]) / 3.0
ψz0[0,0,0] = (ψz0[1,0,0] + ψz0[0,1,0] + ψz0[0,0,1]) / 3.0

ψx0[0,0,nz-1] = (ψx0[0,0,nz-2] + ψx0[1,0,nz-1] + ψx0[0,1,nz-1]) / 3.0                               # Lower back left
ψy0[0,0,nz-1] = (ψy0[0,0,nz-2] + ψy0[1,0,nz-1] + ψy0[0,1,nz-1]) / 3.0
ψz0[0,0,nz-1] = (ψz0[0,0,nz-2] + ψz0[1,0,nz-1] + ψz0[0,1,nz-1]) / 3.0

ψx0[nx-1,0,0] = (ψx0[nx-2,0,0] + ψx0[nx-1,1,0] + ψx0[nx-1,0,1]) / 3.0                               # Lower front right 
ψy0[nx-1,0,0] = (ψy0[nx-2,0,0] + ψy0[nx-1,1,0] + ψy0[nx-1,0,1]) / 3.0
ψz0[nx-1,0,0] = (ψz0[nx-2,0,0] + ψz0[nx-1,1,0] + ψz0[nx-1,0,1]) / 3.0

ψx0[nx-1,0,nz-1] = (ψx0[nx-2,0,nz-1] + ψx0[nx-1,0,nz-2] + ψx0[nx-1,1,nz-1]) / 3.0                   # Lower back right
ψy0[nx-1,0,nz-1] = (ψy0[nx-2,0,nz-1] + ψy0[nx-1,0,nz-2] + ψy0[nx-1,1,nz-1]) / 3.0
ψz0[nx-1,0,nz-1] = (ψz0[nx-2,0,nz-1] + ψz0[nx-1,0,nz-2] + ψz0[nx-1,1,nz-1]) / 3.0

ψx0[0,ny-1,0] = (ψx0[1,ny-1,0] + ψx0[0,ny-2,0] + ψx0[0,ny-1,1]) / 3.0                               # Front top left
ψy0[0,ny-1,0] = (ψy0[1,ny-1,0] + ψy0[0,ny-2,0] + ψy0[0,ny-1,1]) / 3.0
ψz0[0,ny-1,0] = (ψz0[1,ny-1,0] + ψz0[0,ny-2,0] + ψz0[0,ny-1,1]) / 3.0

ψx0[0,ny-1,nz-1] = (ψx0[0,ny-1,nz-2] + ψx0[1,ny-1,nz-1] + ψx0[0,ny-2,nz-1]) / 3.0                   # Back top left
ψy0[0,ny-1,nz-1] = (ψy0[0,ny-1,nz-2] + ψy0[1,ny-1,nz-1] + ψy0[0,ny-2,nz-1]) / 3.0
ψz0[0,ny-1,nz-1] = (ψz0[0,ny-1,nz-2] + ψz0[1,ny-1,nz-1] + ψz0[0,ny-2,nz-1]) / 3.0

ψx0[nx-1,ny-1,0] = (ψx0[nx-2,ny-1,0] + ψx0[nx-1,ny-2,0] + ψx0[nx-1,ny-1,1]) / 3.0                   # Front top right
ψy0[nx-1,ny-1,0] = (ψy0[nx-2,ny-1,0] + ψy0[nx-1,ny-2,0] + ψy0[nx-1,ny-1,1]) / 3.0
ψz0[nx-1,ny-1,0] = (ψz0[nx-2,ny-1,0] + ψz0[nx-1,ny-2,0] + ψz0[nx-1,ny-1,1]) / 3.0

ψx0[nx-1,ny-1,nz-1] = (ψx0[nx-2,ny-1,nz-1] + ψx0[nx-1,ny-1,nz-2] + ψx0[nx-1,ny-2,nz-1]) / 3.0       # Back top right
ψy0[nx-1,ny-1,nz-1] = (ψy0[nx-2,ny-1,nz-1] + ψy0[nx-1,ny-1,nz-2] + ψy0[nx-1,ny-2,nz-1]) / 3.0
ψz0[nx-1,ny-1,nz-1] = (ψz0[nx-2,ny-1,nz-1] + ψz0[nx-1,ny-1,nz-2] + ψz0[nx-1,ny-2,nz-1]) / 3.0

print(f"ψx0 is : ")
print()
print(ψx0)

print(f"Ωx0 is : ")
print()
print(Ωx0)

#--------------------------------------------
# VELOCITY VECTOR FIELD BOUNDARY CONDITIONS
#--------------------------------------------

for j in range(1,ny-1):
    for k in range(1,nz-1):
        u0[0,j,k] = 0.0          # Left wall
        v0[0,j,k] = 0.0
        w0[0,j,k] = 0.0
        u0[nx-1,j,k] = 0.0       # Right wall
        v0[nx-1,j,k] = 0.0
        w0[nx-1,j,k] = 0.0

for i in range(1,nx-1):
    for j in range(1,ny-1):
        u0[i,j,0] = 0.0          # Front wall
        v0[i,j,0] = 0.0
        w0[i,j,0] = 0.0
        u0[i,j,nz-1] = 0.0       # Back wall
        v0[i,j,nz-1] = 0.0
        w0[i,j,nz-1] = 0.0

for k in range(1,nz-1):
    for i in range(1,nx-1):
        u0[i,0,k] = 0.0          # Bottom wall
        v0[i,0,k] = 0.0
        w0[i,0,k] = 0.0
        u0[i,0,k] = Ut          # Top wall
        v0[i,0,k] = 0.0
        w0[i,0,k] = 0.0

# Velocity edge points

for j in range(1,ny-1):
    u0[0,j,0] = (u0[1,j,0] + u0[0,j,1])/2.0                            # Front-left edge
    v0[0,j,0] = (v0[1,j,0] + v0[0,j,1])/2.0
    w0[0,j,0] = (w0[1,j,0] + w0[0,j,1])/2.0
    u0[nx-1,j,0] = (u0[nx-2,j,0] + u0[nx-1,j,1])/2.0                   # Front-right edge
    v0[nx-1,j,0] = (v0[nx-2,j,0] + v0[nx-1,j,1])/2.0
    w0[nx-1,j,0] = (w0[nx-2,j,0] + w0[nx-1,j,1])/2.0
    u0[nx-1,j,nz-1] = (u0[nx-2,j,nz-1] + u0[nx-1,j,nz-2])/2.0          # Back-right edge
    v0[nx-1,j,nz-1] = (v0[nx-2,j,nz-1] + v0[nx-1,j,nz-2])/2.0
    w0[nx-1,j,nz-1] = (w0[nx-2,j,nz-1] + w0[nx-1,j,nz-2])/2.0
    u0[0,j,nz-1] = (u0[1,j,nz-1] + u0[0,j,nz-2])/2.0                   # Back-left edge
    v0[0,j,nz-1] = (v0[1,j,nz-1] + v0[0,j,nz-2])/2.0
    w0[0,j,nz-1] = (w0[1,j,nz-1] + w0[0,j,nz-2])/2.0

for z in range(1,nz-1):
    u0[0,0,z] = (u0[1,0,z] + u0[0,1,z])/2.0                            # Bottom-left edge
    v0[0,0,z] = (v0[1,0,z] + v0[0,1,z])/2.0        
    w0[0,0,z] = (w0[1,0,z] + w0[0,1,z])/2.0
    u0[nx-1,0,z] = (u0[nx-2,0,z] + u0[0,1,z])/2.0                      # Bottom-right edge
    v0[nx-1,0,z] = (v0[nx-2,0,z] + v0[0,1,z])/2.0        
    w0[nx-1,0,z] = (w0[nx-2,0,z] + w0[0,1,z])/2.0
    u0[nx-1,ny-1,z] = (u0[nx-2,ny-1,z] + u0[nx-1,ny-2,z])/2.0          # Top-right edge
    v0[nx-1,ny-1,z] = (v0[nx-2,ny-1,z] + v0[nx-1,ny-2,z])/2.0        
    w0[nx-1,ny-1,z] = (w0[nx-2,ny-1,z] + w0[nx-1,ny-2,z])/2.0
    u0[0,ny-1,z] = (u0[0,ny-2,z] + u0[1,ny-1,z])/2.0                   # Top-left edge
    v0[0,ny-1,z] = (v0[0,ny-2,z] + v0[1,ny-1,z])/2.0        
    w0[0,ny-1,z] = (w0[0,ny-2,z] + w0[1,ny-1,z])/2.0

for i in range(1,nx-1):
    u0[i,0,0] = (u0[i,1,0] + u0[i,0,1])/2.0                            # Front-bottom edge
    v0[i,0,0] = (v0[i,1,0] + v0[i,0,1])/2.0        
    w0[i,0,0] = (w0[i,1,0] + w0[i,0,1])/2.0
    u0[i,0,nz-1] = (u0[i,1,nz-1] + u0[i,0,nz-2])/2.0                   # Back-bottom edge
    v0[i,0,nz-1] = (v0[i,1,nz-1] + v0[i,0,nz-2])/2.0         
    w0[i,0,nz-1] = (w0[i,1,nz-1] + w0[i,0,nz-2])/2.0 
    u0[i,ny-1,0] = (u0[i,ny-1,1] + u0[i,ny-2,0])/2.0                   # Front-top edge
    v0[i,ny-1,0] = (v0[i,ny-1,1] + v0[i,ny-2,0])/2.0         
    w0[i,ny-1,0] = (w0[i,ny-1,1] + w0[i,ny-2,0])/2.0 
    u0[i,ny-1,nz-1] = (u0[i,ny-2,nz-1] + u0[i,ny-1,nz-2])/2.0          # Back-top edge
    v0[i,ny-1,nz-1] = (v0[0,ny-2,nz-1] + v0[1,ny-1,nz-2])/2.0        
    w0[i,ny-1,nz-1] = (w0[0,ny-2,nz-1] + w0[1,ny-1,nz-2])/2.0

# Velocity corner points
u0[0,0,0] = (u0[1,0,0] + u0[0,1,0] + u0[0,0,1]) / 3.0                                           # Lower bottom left 
v0[0,0,0] = (v0[1,0,0] + v0[0,1,0] + v0[0,0,1]) / 3.0
w0[0,0,0] = (w0[1,0,0] + w0[0,1,0] + w0[0,0,1]) / 3.0

u0[0,0,nz-1] = (u0[0,0,nz-2] + u0[1,0,nz-1] + u0[0,1,nz-1]) / 3.0                               # Lower back left
v0[0,0,nz-1] = (v0[0,0,nz-2] + v0[1,0,nz-1] + v0[0,1,nz-1]) / 3.0
w0[0,0,nz-1] = (w0[0,0,nz-2] + w0[1,0,nz-1] + w0[0,1,nz-1]) / 3.0

u0[nx-1,0,0] = (u0[nx-2,0,0] + u0[nx-1,1,0] + u0[nx-1,0,1]) / 3.0                               # Lower front right 
v0[nx-1,0,0] = (v0[nx-2,0,0] + v0[nx-1,1,0] + v0[nx-1,0,1]) / 3.0
w0[nx-1,0,0] = (w0[nx-2,0,0] + w0[nx-1,1,0] + w0[nx-1,0,1]) / 3.0

u0[nx-1,0,nz-1] = (u0[nx-2,0,nz-1] + u0[nx-1,0,nz-2] + u0[nx-1,1,nz-1]) / 3.0                   # Lower back right
v0[nx-1,0,nz-1] = (v0[nx-2,0,nz-1] + v0[nx-1,0,nz-2] + v0[nx-1,1,nz-1]) / 3.0
w0[nx-1,0,nz-1] = (w0[nx-2,0,nz-1] + w0[nx-1,0,nz-2] + w0[nx-1,1,nz-1]) / 3.0

u0[0,ny-1,0] = (u0[1,ny-1,0] + u0[0,ny-2,0] + u0[0,ny-1,1]) / 3.0                               # Front top left
v0[0,ny-1,0] = (v0[1,ny-1,0] + v0[0,ny-2,0] + v0[0,ny-1,1]) / 3.0
w0[0,ny-1,0] = (w0[1,ny-1,0] + w0[0,ny-2,0] + w0[0,ny-1,1]) / 3.0

u0[0,ny-1,nz-1] = (u0[0,ny-1,nz-2] + u0[1,ny-1,nz-1] + u0[0,ny-2,nz-1]) / 3.0                   # Back top left
v0[0,ny-1,nz-1] = (v0[0,ny-1,nz-2] + v0[1,ny-1,nz-1] + v0[0,ny-2,nz-1]) / 3.0
w0[0,ny-1,nz-1] = (w0[0,ny-1,nz-2] + w0[1,ny-1,nz-1] + w0[0,ny-2,nz-1]) / 3.0

u0[nx-1,ny-1,0] = (u0[nx-2,ny-1,0] + u0[nx-1,ny-2,0] + u0[nx-1,ny-1,1]) / 3.0                   # Front top right
v0[nx-1,ny-1,0] = (v0[nx-2,ny-1,0] + v0[nx-1,ny-2,0] + v0[nx-1,ny-1,1]) / 3.0
w0[nx-1,ny-1,0] = (w0[nx-2,ny-1,0] + w0[nx-1,ny-2,0] + w0[nx-1,ny-1,1]) / 3.0

u0[nx-1,ny-1,nz-1] = (u0[nx-2,ny-1,nz-1] + u0[nx-1,ny-1,nz-2] + u0[nx-1,ny-2,nz-1]) / 3.0       # Back top right
v0[nx-1,ny-1,nz-1] = (v0[nx-2,ny-1,nz-1] + v0[nx-1,ny-1,nz-2] + v0[nx-1,ny-2,nz-1]) / 3.0
w0[nx-1,ny-1,nz-1] = (w0[nx-2,ny-1,nz-1] + w0[nx-1,ny-1,nz-2] + w0[nx-1,ny-2,nz-1]) / 3.0


# If vorticity boundary conditions are using vorticity, we also need the initial velocity field
# and boundary conditions?

#----------------------------------------------
# VORTICITY VECTOR FIELD (Ω) BOUNDARY CONDITIONS
#----------------------------------------------

for j in range(1,ny-1):
    for k in range(1,nz-1):
        Ωx0[0,j,k] = 0.0                                 # Left wall
        Ωy0[0,j,k] = -w0[1,j,k]/dx
        Ωz0[0,j,k] = v0[1,j,k]/dx
        Ωx0[nx-1,j,k] = 0.0                              # Right wall
        Ωy0[nx-1,j,k] = -w0[nx-2,j,k]/dx
        Ωz0[nx-1,j,k] =-v0[nx-2,j,k]/dx

for i in range(1,nx-1):
    for j in range(1,ny-1):
        Ωx0[i,j,0] = -v0[i,j,1]/dz                        # Front wall
        Ωy0[i,j,0] = u0[i,j,1]/dz
        Ωz0[i,j,0] = 0.0
        Ωx0[i,j,nz-1] = v0[i,j,nz-2]/dz                   # Back wall
        Ωy0[i,j,nz-1] = -u0[i,j,nz-2]/dz
        Ωz0[i,j,nz-1] = 0.0
        
for i in range(1,nx-1):
    for k in range(1,nz-1):
        Ωx0[i,0,k] = w0[i,1,k]/dy                         # Bottom wall
        Ωy0[i,0,k] = 0.0
        Ωz0[i,0,k] = -u0[i,1,k]/dy
        Ωx0[i,0,k] = -w0[i,ny-2,k]/dy                     # Top wall
        Ωy0[i,0,k] = (u0[i,ny-1,k+1] - u0[i,ny-1,k])/dy
        Ωz0[i,0,k] = -(u0[i,ny-1,k] - u0[i,ny-2,k])/dy

# Vorticity edge points

# Vorticity corner points
Ωx0[0,0,0] = (Ωx0[1,0,0] + Ωx0[0,1,0] + Ωx0[0,0,1]) / 3.0
Ωy0[0,0,0] = (Ωy0[1,0,0] + Ωy0[0,1,0] + Ωy0[0,0,1]) / 3.0
Ωz0[0,0,0] = (Ωz0[1,0,0] + Ωz0[0,1,0] + Ωz0[0,0,1]) / 3.0

Ωx0[0,0,nz-1] = (Ωx0[0,0,nz-2] + Ωx0[1,0,nz-1] + Ωx0[0,1,nz-1]) / 3.0
Ωy0[0,0,nz-1] = (Ωy0[0,0,nz-2] + Ωy0[1,0,nz-1] + Ωy0[0,1,nz-1]) / 3.0
Ωz0[0,0,nz-1] = (Ωz0[0,0,nz-2] + Ωz0[1,0,nz-1] + Ωz0[0,1,nz-1]) / 3.0

Ωx0[nx-1,0,0] = (Ωx0[nx-2,0,0] + Ωx0[nx-1,1,0] + Ωx0[nx-1,0,1]) / 3.0
Ωy0[nx-1,0,0] = (Ωy0[nx-2,0,0] + Ωy0[nx-1,1,0] + Ωy0[nx-1,0,1]) / 3.0
ψz0[nx-1,0,0] = (ψz0[nx-2,0,0] + ψz0[nx-1,1,0] + ψz0[nx-1,0,1]) / 3.0

Ωx0[nx-1,0,nz-1] = (Ωx0[nx-2,0,nz-1] + Ωx0[nx-1,0,nz-2] + Ωx0[nx-1,1,nz-1]) / 3.0
Ωy0[nx-1,0,nz-1] = (Ωy0[nx-2,0,nz-1] + Ωy0[nx-1,0,nz-2] + Ωy0[nx-1,1,nz-1]) / 3.0
Ωz0[nx-1,0,nz-1] = (Ωz0[nx-2,0,nz-1] + Ωz0[nx-1,0,nz-2] + Ωz0[nx-1,1,nz-1]) / 3.0

Ωx0[0,ny-1,0] = (Ωx0[1,ny-1,0] + Ωx0[0,ny-2,0] + Ωx0[0,ny-1,1]) / 3.0
Ωy0[0,ny-1,0] = (Ωy0[1,ny-1,0] + Ωy0[0,ny-2,0] + Ωy0[0,ny-1,1]) / 3.0
Ωz0[0,ny-1,0] = (Ωz0[1,ny-1,0] + Ωz0[0,ny-2,0] + Ωz0[0,ny-1,1]) / 3.0

Ωx0[0,ny-1,nz-1] = (Ωx0[0,ny-1,nz-2] + Ωx0[1,ny-1,nz-1] + Ωx0[0,ny-2,nz-1]) / 3.0
Ωy0[0,ny-1,nz-1] = (Ωy0[0,ny-1,nz-2] + Ωy0[1,ny-1,nz-1] + Ωy0[0,ny-2,nz-1]) / 3.0
Ωz0[0,ny-1,nz-1] = (Ωz0[0,ny-1,nz-2] + Ωz0[1,ny-1,nz-1] + Ωz0[0,ny-2,nz-1]) / 3.0

Ωx0[nx-1,ny-1,0] = (Ωx0[nx-2,ny-1,0] + Ωx0[nx-1,ny-2,0] + Ωx0[nx-1,ny-1,1]) / 3.0
Ωy0[nx-1,ny-1,0] = (Ωy0[nx-2,ny-1,0] + Ωy0[nx-1,ny-2,0] + Ωy0[nx-1,ny-1,1]) / 3.0
Ωz0[nx-1,ny-1,0] = (Ωz0[nx-2,ny-1,0] + Ωz0[nx-1,ny-2,0] + Ωz0[nx-1,ny-1,1]) / 3.0

Ωx0[nx-1,ny-1,nz-1] = (Ωx0[nx-2,ny-1,nz-1] + Ωx0[nx-1,ny-1,nz-2] + Ωx0[nx-1,ny-2,nz-1]) / 3.0
Ωy0[nx-1,ny-1,nz-1] = (Ωy0[nx-2,ny-1,nz-1] + Ωy0[nx-1,ny-1,nz-2] + Ωy0[nx-1,ny-2,nz-1]) / 3.0
Ωz0[nx-1,ny-1,nz-1] = (Ωz0[nx-2,ny-1,nz-1] + Ωz0[nx-1,ny-1,nz-2] + Ωz0[nx-1,ny-2,nz-1]) / 3.0


# Create solution storage
ψx_sol = []
ψx_sol.append(ψx0)
ψy_sol = []
ψy_sol.append(ψy0)
ψz_sol = []
ψz_sol.append(ψz0)

Ωx_sol = []
Ωx_sol.append(Ωx0)
Ωy_sol = []
Ωy_sol.append(Ωy0)
Ωz_sol = []
Ωz_sol.append(Ωz0)

u_sol = []
u_sol.append(u0)
v_sol = []
v_sol.append(v0)
w_sol = []
w_sol.append(w0)



# Time-marching parameters
tend = 5.0
tol = 1e-3
err = 1e5
itmax = 100
β = 1.7
dt = min(0.25*dx*dx/Ϟ, 4*Ϟ/Ut/Ut)

# Start main time loop
t = 0
while t < tend:

    #------------------------------------------------------------------
    # SOLVE THREE VECTOR-POTENTIAL POISSON EQUATIONS USING ITERATION
    #------------------------------------------------------------------

    # POISSON SOLVER FOR ψ_x
    it = 0
    ψx = ψx_sol[-1].copy()
    Ωx = Ωx_sol[-1].copy()
    while it < itmax and err > tol:
        ψx_k = ψx.copy()
        for i in range(1,nx-1):
            for j in range(1,ny-1):
                for k in range(1,nz-1):
                    ψx[i,j,k] = (β / (2*(dx**2*dz**2 + dy**2*dz**2 + dx**2*dy**2))) * (dx**2*dy**2*dz**2*Ωx[i,j,k] + dy**2*dz**2*(ψx[i+1,j,k]+ψx[i-1,j,k]) + dx**2*dz**2*(ψx[i,j+1,k]+ψx[i,j-1,k]) + dx**2*dy**2*(ψx[i,j,k+1] + ψx[i,j,k+1])) + (1 - β) * ψx[i,j,k]
        err = np.linalg.norm(ψx.ravel() - ψx_k.ravel())
        it = it + 1
    ψx_sol.append(ψx)

    # POISSON SOLVER FOR ψ_y
    it = 0
    ψy = ψy_sol[-1].copy()
    Ωy = Ωy_sol[-1].copy()
    while it < itmax and err > tol:
        ψy_k = ψy.copy()
        for i in range(1,nx-1):
            for j in range(1,ny-1):
                for k in range(1,nz-1):
                    ψy[i,j,k] = (β / (2*(dx**2*dz**2 + dy**2*dz**2 + dx**2*dy**2))) * (dx**2*dy**2*dz**2*Ωy[i,j,k] + dy**2*dz**2*(ψy[i+1,j,k]+ψy[i-1,j,k]) + dx**2*dz**2*(ψy[i,j+1,k]+ψy[i,j-1,k]) + dx**2*dy**2*(ψy[i,j,k+1] + ψy[i,j,k+1])) + (1 - β) * ψy[i,j,k]
        err = np.linalg.norm(ψy.ravel() - ψy_k.ravel())
        it = it + 1
    ψy_sol.append(ψy)

    # POISSON SOLVER FOR ψ_z
    it = 0 
    ψz = ψz_sol[-1].copy()
    Ωz = Ωz_sol[-1].copy()
    while it < itmax and err > tol:
        ψz_k = ψz.copy()
        for i in range(1,nx-1):
            for j in range(1,ny-1):
                for k in range(1,nz-1):
                    ψz[i,j,k] = (β / (2*(dx**2*dz**2 + dy**2*dz**2 + dx**2*dy**2))) * (dx**2*dy**2*dz**2*Ωz[i,j,k] + dy**2*dz**2*(ψz[i+1,j,k]+ψz[i-1,j,k]) + dx**2*dz**2*(ψz[i,j+1,k]+ψz[i,j-1,k]) + dx**2*dy**2*(ψz[i,j,k+1] + ψz[i,j,k+1])) + (1 - β) * ψz[i,j,k]
        err = np.linalg.norm(ψz.ravel() - ψz_k.ravel())
        it = it + 1
    ψz_sol.append(ψz)




    #---------------------------------------
    # SOLVE FOR THE VELOCITY VECTOR FIELD
    #---------------------------------------
    # Use the above calculated ψ field in several finite differences for the velocity components
    # And enforce velocity boundary conditions
    # This is uses the definition of the 3D velocity vector field, which is the curl of the vector-potential field
    for i in range(1,nx-1):
        for j in range(1,ny-1):
            for k in range(1,nz-1):
                u[i,j,k] = (ψ[i+1,j,k] - ψ[i,j,k])/dx
                v[i,j,k] = (ψ[i,j+1,k] - ψ[i,j,k])/dy
                w[i,j,k] = (ψ[i,j,k+1] - ψ[i,j,k])/dz






    #---------------------------------------------------------------------------------
    # SOLVE THE 3D VORTICITY TRANSPORT EQUATION INCLUDING THE VORTEX STRETCHING TERM
    #---------------------------------------------------------------------------------
    # We solve three equations, one for each vorticity component

    #Ω_sol.append(Ω)





    t = t + dt


