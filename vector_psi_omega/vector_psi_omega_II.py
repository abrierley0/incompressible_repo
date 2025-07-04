import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(linewidth=1000, threshold=np.inf)  # Ensure full matrix prints

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
nu = 0.05
Ut = 3.2
Re = Ut*Lx/nu
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
    for k in range(1,nz-1):
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
    ψx0[0,j,0] = (ψx0[1,j,0] + ψx0[0,j,1])/2.0                            # Front-left edge
    ψy0[0,j,0] = (ψy0[1,j,0] + ψy0[0,j,1])/2.0
    ψz0[0,j,0] = (ψz0[1,j,0] + ψz0[0,j,1])/2.0
    ψx0[nx-1,j,0] = (ψx0[nx-2,j,0] + ψx0[nx-1,j,1])/2.0                   # Front-right edge
    ψy0[nx-1,j,0] = (ψy0[nx-2,j,0] + ψy0[nx-1,j,1])/2.0
    ψz0[nx-1,j,0] = (ψz0[nx-2,j,0] + ψz0[nx-1,j,1])/2.0
    ψx0[nx-1,j,nz-1] = (ψx0[nx-2,j,nz-1] + ψx0[nx-1,j,nz-2])/2.0          # Back-right edge
    ψy0[nx-1,j,nz-1] = (ψy0[nx-2,j,nz-1] + ψy0[nx-1,j,nz-2])/2.0
    ψz0[nx-1,j,nz-1] = (ψz0[nx-2,j,nz-1] + ψz0[nx-1,j,nz-2])/2.0
    ψx0[0,j,nz-1] = (ψx0[1,j,nz-1] + ψx0[0,j,nz-2])/2.0                   # Back-left edge
    ψy0[0,j,nz-1] = (ψy0[1,j,nz-1] + ψy0[0,j,nz-2])/2.0
    ψz0[0,j,nz-1] = (ψz0[1,j,nz-1] + ψz0[0,j,nz-2])/2.0

for z in range(1,nz-1):
    ψx0[0,0,z] = (ψx0[1,0,z] + ψx0[0,1,z])/2.0                            # Bottom-left edge
    ψy0[0,0,z] = (ψy0[1,0,z] + ψy0[0,1,z])/2.0        
    ψz0[0,0,z] = (ψz0[1,0,z] + ψz0[0,1,z])/2.0
    ψx0[nx-1,0,z] = (ψx0[nx-2,0,z] + ψx0[0,1,z])/2.0                      # Bottom-right edge
    ψy0[nx-1,0,z] = (ψy0[nx-2,0,z] + ψy0[0,1,z])/2.0        
    ψz0[nx-1,0,z] = (ψz0[nx-2,0,z] + ψz0[0,1,z])/2.0
    ψx0[nx-1,ny-1,z] = (ψx0[nx-2,ny-1,z] + ψx0[nx-1,ny-2,z])/2.0          # Top-right edge
    ψy0[nx-1,ny-1,z] = (ψy0[nx-2,ny-1,z] + ψy0[nx-1,ny-2,z])/2.0        
    ψz0[nx-1,ny-1,z] = (ψz0[nx-2,ny-1,z] + ψz0[nx-1,ny-2,z])/2.0
    ψx0[0,ny-1,z] = (ψx0[0,ny-2,z] + ψx0[1,ny-1,z])/2.0                   # Top-left edge
    ψy0[0,ny-1,z] = (ψy0[0,ny-2,z] + ψy0[1,ny-1,z])/2.0        
    ψz0[0,ny-1,z] = (ψz0[0,ny-2,z] + ψz0[1,ny-1,z])/2.0

for i in range(1,nx-1):
    ψx0[i,0,0] = (ψx0[i,1,0] + ψx0[i,0,1])/2.0                            # Front-bottom edge
    ψy0[i,0,0] = (ψy0[i,1,0] + ψy0[i,0,1])/2.0        
    ψz0[i,0,0] = (ψz0[i,1,0] + ψz0[i,0,1])/2.0
    ψx0[i,0,nz-1] = (ψx0[i,1,nz-1] + ψx0[i,0,nz-2])/2.0                   # Back-bottom edge
    ψy0[i,0,nz-1] = (ψy0[i,1,nz-1] + ψy0[i,0,nz-2])/2.0         
    ψz0[i,0,nz-1] = (ψz0[i,1,nz-1] + ψz0[i,0,nz-2])/2.0 
    ψx0[i,ny-1,0] = (ψx0[i,ny-1,1] + ψx0[i,ny-2,0])/2.0                   # Front-top edge
    ψy0[i,ny-1,0] = (ψy0[i,ny-1,1] + ψy0[i,ny-2,0])/2.0         
    ψz0[i,ny-1,0] = (ψz0[i,ny-1,1] + ψz0[i,ny-2,0])/2.0 
    ψx0[i,ny-1,nz-1] = (ψx0[i,ny-2,nz-1] + ψx0[i,ny-1,nz-2])/2.0          # Back-top edge
    ψy0[i,ny-1,nz-1] = (ψy0[0,ny-2,nz-1] + ψy0[1,ny-1,nz-2])/2.0        
    ψz0[i,ny-1,nz-1] = (ψz0[0,ny-2,nz-1] + ψz0[1,ny-1,nz-2])/2.0

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

print()
print(f"ψx0 is : ")
print()
print(ψx0)
print()


#--------------------------------------------
# VELOCITY VECTOR FIELD BOUNDARY CONDITIONS
#--------------------------------------------
# currently, just using no-slip and impermeability
# conditions directly applied to u,v,w. But may need to use the
# definition

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
        u0[i,ny-1,k] = Ut          # Top wall
        v0[i,ny-1,k] = 0.0
        w0[i,ny-1,k] = 0.0

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

print()
print(f"u0 is : ")
print()
print(u0)
print()



#------------------------------------------------
# VORTICITY VECTOR FIELD (Ω) BOUNDARY CONDITIONS
#------------------------------------------------
# derived using the definition of vorticity

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
        Ωx0[i,ny-1,k] = -w0[i,ny-2,k]/dy                     # Top wall
        Ωy0[i,ny-1,k] = (u0[i,ny-1,k+1] - u0[i,ny-1,k])/dy
        Ωz0[i,ny-1,k] = -(u0[i,ny-1,k] - u0[i,ny-2,k])/dy

# Vorticity edge points
for j in range(1,ny-1):
    Ωx0[0,j,0] = (Ωx0[1,j,0] + Ωx0[0,j,1])/2.0                            # Front-left edge
    Ωy0[0,j,0] = (Ωy0[1,j,0] + Ωy0[0,j,1])/2.0
    Ωz0[0,j,0] = (Ωz0[1,j,0] + Ωz0[0,j,1])/2.0
    Ωx0[nx-1,j,0] = (Ωx0[nx-2,j,0] + Ωx0[nx-1,j,1])/2.0                   # Front-right edge
    Ωy0[nx-1,j,0] = (Ωy0[nx-2,j,0] + Ωy0[nx-1,j,1])/2.0
    Ωz0[nx-1,j,0] = (Ωz0[nx-2,j,0] + Ωz0[nx-1,j,1])/2.0
    Ωx0[nx-1,j,nz-1] = (Ωx0[nx-2,j,nz-1] + Ωx0[nx-1,j,nz-2])/2.0          # Back-right edge
    Ωy0[nx-1,j,nz-1] = (Ωy0[nx-2,j,nz-1] + Ωy0[nx-1,j,nz-2])/2.0
    Ωz0[nx-1,j,nz-1] = (Ωz0[nx-2,j,nz-1] + Ωz0[nx-1,j,nz-2])/2.0
    Ωx0[0,j,nz-1] = (Ωx0[1,j,nz-1] + Ωx0[0,j,nz-2])/2.0                   # Back-left edge
    Ωy0[0,j,nz-1] = (Ωy0[1,j,nz-1] + Ωy0[0,j,nz-2])/2.0
    Ωz0[0,j,nz-1] = (Ωz0[1,j,nz-1] + Ωz0[0,j,nz-2])/2.0

for z in range(1,nz-1):
    Ωx0[0,0,z] = (Ωx0[1,0,z] + Ωx0[0,1,z])/2.0                            # Bottom-left edge
    Ωy0[0,0,z] = (Ωy0[1,0,z] + Ωy0[0,1,z])/2.0        
    Ωz0[0,0,z] = (Ωz0[1,0,z] + Ωz0[0,1,z])/2.0
    Ωx0[nx-1,0,z] = (Ωx0[nx-2,0,z] + Ωx0[0,1,z])/2.0                      # Bottom-right edge
    Ωy0[nx-1,0,z] = (Ωy0[nx-2,0,z] + Ωy0[0,1,z])/2.0        
    Ωz0[nx-1,0,z] = (Ωz0[nx-2,0,z] + Ωz0[0,1,z])/2.0
    Ωx0[nx-1,ny-1,z] = (Ωx0[nx-2,ny-1,z] + Ωx0[nx-1,ny-2,z])/2.0          # Top-right edge
    Ωy0[nx-1,ny-1,z] = (Ωy0[nx-2,ny-1,z] + Ωy0[nx-1,ny-2,z])/2.0        
    Ωz0[nx-1,ny-1,z] = (Ωz0[nx-2,ny-1,z] + Ωz0[nx-1,ny-2,z])/2.0
    Ωx0[0,ny-1,z] = (Ωx0[0,ny-2,z] + Ωx0[1,ny-1,z])/2.0                   # Top-left edge
    Ωy0[0,ny-1,z] = (Ωy0[0,ny-2,z] + Ωy0[1,ny-1,z])/2.0        
    Ωz0[0,ny-1,z] = (Ωz0[0,ny-2,z] + Ωz0[1,ny-1,z])/2.0

for i in range(1,nx-1):
    Ωx0[i,0,0] = (Ωx0[i,1,0] + Ωx0[i,0,1])/2.0                            # Front-bottom edge
    Ωy0[i,0,0] = (Ωy0[i,1,0] + Ωy0[i,0,1])/2.0        
    Ωz0[i,0,0] = (Ωz0[i,1,0] + Ωz0[i,0,1])/2.0
    Ωx0[i,0,nz-1] = (Ωx0[i,1,nz-1] + Ωx0[i,0,nz-2])/2.0                   # Back-bottom edge
    Ωy0[i,0,nz-1] = (Ωy0[i,1,nz-1] + Ωy0[i,0,nz-2])/2.0         
    Ωz0[i,0,nz-1] = (Ωz0[i,1,nz-1] + Ωz0[i,0,nz-2])/2.0 
    Ωx0[i,ny-1,0] = (Ωx0[i,ny-1,1] + Ωx0[i,ny-2,0])/2.0                   # Front-top edge
    Ωy0[i,ny-1,0] = (Ωy0[i,ny-1,1] + Ωy0[i,ny-2,0])/2.0         
    Ωz0[i,ny-1,0] = (Ωz0[i,ny-1,1] + Ωz0[i,ny-2,0])/2.0 
    Ωx0[i,ny-1,nz-1] = (Ωx0[i,ny-2,nz-1] + Ωx0[i,ny-1,nz-2])/2.0          # Back-top edge
    Ωy0[i,ny-1,nz-1] = (Ωy0[0,ny-2,nz-1] + Ωy0[1,ny-1,nz-2])/2.0        
    Ωz0[i,ny-1,nz-1] = (Ωz0[0,ny-2,nz-1] + Ωz0[1,ny-1,nz-2])/2.0

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

print()
print(f"Ωx0 is : ")
print()
print(Ωx0)
print()


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
errx = 1e5
erry = 1e5
errz = 1e5
itmax = 100
β = 1.7
dt = min(0.25*dx*dx/nu, 4*nu/Ut/Ut)

# Start main time loop
t = 0
u = u0.copy()
v = v0.copy()
w = w0.copy()
while t < tend:

    #------------------------------------------------------------------
    # SOLVE THREE VECTOR-POTENTIAL POISSON EQUATIONS USING ITERATION
    #------------------------------------------------------------------

    # POISSON SOLVER FOR ψ_x
    it = 0
    ψx = ψx_sol[-1].copy()
    Ωxn = Ωx_sol[-1].copy()
    while it < itmax and errx > tol:
        ψx_k = ψx.copy()
        for i in range(1,nx-1):
            for j in range(1,ny-1):
                for k in range(1,nz-1):
                    ψx[i,j,k] = (β / (2*(dx**2*dz**2 + dy**2*dz**2 + dx**2*dy**2))) * (dx**2*dy**2*dz**2*Ωxn[i,j,k] + dy**2*dz**2*(ψx[i+1,j,k]+ψx[i-1,j,k]) + dx**2*dz**2*(ψx[i,j+1,k]+ψx[i,j-1,k]) + dx**2*dy**2*(ψx[i,j,k+1] + ψx[i,j,k+1])) + (1 - β) * ψx[i,j,k]
        errx = np.linalg.norm(ψx.ravel() - ψx_k.ravel())
        it = it + 1
        print(f"X Iteration: {it}")
        print(f"X Error: {errx}")

    # POISSON SOLVER FOR ψ_y
    it = 0
    ψy = ψy_sol[-1].copy()
    Ωyn = Ωy_sol[-1].copy()
    while it < itmax and erry > tol:
        ψy_k = ψy.copy()
        for i in range(1,nx-1):
            for j in range(1,ny-1):
                for k in range(1,nz-1):
                    ψy[i,j,k] = (β / (2*(dx**2*dz**2 + dy**2*dz**2 + dx**2*dy**2))) * (dx**2*dy**2*dz**2*Ωyn[i,j,k] + dy**2*dz**2*(ψy[i+1,j,k]+ψy[i-1,j,k]) + dx**2*dz**2*(ψy[i,j+1,k]+ψy[i,j-1,k]) + dx**2*dy**2*(ψy[i,j,k+1] + ψy[i,j,k+1])) + (1 - β) * ψy[i,j,k]
        erry = np.linalg.norm(ψy.ravel() - ψy_k.ravel())
        it = it + 1
        print(f"Y Iteration: {it}")
        print(f"Y Error: {erry}")

    # POISSON SOLVER FOR ψ_z
    it = 0 
    ψz = ψz_sol[-1].copy()
    Ωzn = Ωz_sol[-1].copy()
    while it < itmax and errz > tol:
        ψz_k = ψz.copy()
        for i in range(1,nx-1):
            for j in range(1,ny-1):
                for k in range(1,nz-1):
                    ψz[i,j,k] = (β / (2*(dx**2*dz**2 + dy**2*dz**2 + dx**2*dy**2))) * (dx**2*dy**2*dz**2*Ωzn[i,j,k] + dy**2*dz**2*(ψz[i+1,j,k]+ψz[i-1,j,k]) + dx**2*dz**2*(ψz[i,j+1,k]+ψz[i,j-1,k]) + dx**2*dy**2*(ψz[i,j,k+1] + ψz[i,j,k+1])) + (1 - β) * ψz[i,j,k]
        errz = np.linalg.norm(ψz.ravel() - ψz_k.ravel())
        it = it + 1
        print(f"Z Iteration: {it}")
        print(f"Z Error: {errz}")


    #-------------------------------------------
    # Re-apply boundary conditions to ψ
    #-------------------------------------------
    for j in range(1,ny-1):
        for k in range(1,nz-1):
            ψx[0,j,k] = ψx[1,j,k]         # Left wall
            ψy[0,j,k] = 0.0
            ψz[0,j,k] = 0.0  
            ψx[nx-1,j,k] = ψx[nx-2,j,k]   # Right wall
            ψy[nx-1,j,k] = 0.0
            ψz[nx-1,j,k] = 0.0

    for i in range(1,nx-1):
        for j in range(1,ny-1):
            ψx[i,j,0] = 0.0                # Front wall
            ψy[i,j,0] = 0.0
            ψz[i,j,0] = ψz[i,j,1]       
            ψx[i,j,nz-1] = 0.0             # Back wall
            ψy[i,j,nz-1] = 0.0
            ψz[i,j,nz-1] = ψz[i,j,nz-2]

    for i in range(1,nx-1):
        for k in range(1,ny-1):
            ψx[i,0,k] = 0.0                # Bottom wall
            ψy[i,0,k] = ψy[i,1,k]   
            ψz[i,0,k] = 0.0
            
            #----------------------------
            # TOP WALL BOUNDARY CONDITION
            #---------------------------
            # NOTE: Check top wall derivation
            ψx[i,ny-1,k] = Ut                # Top wall
            ψy[i,ny-1,k] = ψy[i,ny-2,k]   
            ψz[i,ny-1,k] = 0.0


    # Vector-potential edge points
    for j in range(1,ny-1):
        ψx[0,j,0] = (ψx[1,j,0] + ψx[0,j,1])/2.0                            # Front-left edge
        ψy[0,j,0] = (ψy[1,j,0] + ψy[0,j,1])/2.0
        ψz[0,j,0] = (ψz[1,j,0] + ψz[0,j,1])/2.0
        ψx[nx-1,j,0] = (ψx[nx-2,j,0] + ψx[nx-1,j,1])/2.0                   # Front-right edge
        ψy[nx-1,j,0] = (ψy[nx-2,j,0] + ψy[nx-1,j,1])/2.0
        ψz[nx-1,j,0] = (ψz[nx-2,j,0] + ψz[nx-1,j,1])/2.0
        ψx[nx-1,j,nz-1] = (ψx[nx-2,j,nz-1] + ψx[nx-1,j,nz-2])/2.0          # Back-right edge
        ψy[nx-1,j,nz-1] = (ψy[nx-2,j,nz-1] + ψy[nx-1,j,nz-2])/2.0
        ψz[nx-1,j,nz-1] = (ψz[nx-2,j,nz-1] + ψz[nx-1,j,nz-2])/2.0
        ψx[0,j,nz-1] = (ψx[1,j,nz-1] + ψx[0,j,nz-2])/2.0                   # Back-left edge
        ψy[0,j,nz-1] = (ψy[1,j,nz-1] + ψy[0,j,nz-2])/2.0
        ψz[0,j,nz-1] = (ψz[1,j,nz-1] + ψz[0,j,nz-2])/2.0

    for z in range(1,nz-1):
        ψx[0,0,z] = (ψx[1,0,z] + ψx[0,1,z])/2.0                            # Bottom-left edge
        ψy[0,0,z] = (ψy[1,0,z] + ψy[0,1,z])/2.0        
        ψz[0,0,z] = (ψz[1,0,z] + ψz[0,1,z])/2.0
        ψx[nx-1,0,z] = (ψx[nx-2,0,z] + ψx[0,1,z])/2.0                      # Bottom-right edge
        ψy[nx-1,0,z] = (ψy[nx-2,0,z] + ψy[0,1,z])/2.0        
        ψz[nx-1,0,z] = (ψz[nx-2,0,z] + ψz[0,1,z])/2.0
        ψx[nx-1,ny-1,z] = (ψx[nx-2,ny-1,z] + ψx[nx-1,ny-2,z])/2.0          # Top-right edge
        ψy[nx-1,ny-1,z] = (ψy[nx-2,ny-1,z] + ψy[nx-1,ny-2,z])/2.0        
        ψz[nx-1,ny-1,z] = (ψz[nx-2,ny-1,z] + ψz[nx-1,ny-2,z])/2.0
        ψx[0,ny-1,z] = (ψx[0,ny-2,z] + ψx[1,ny-1,z])/2.0                   # Top-left edge
        ψy[0,ny-1,z] = (ψy[0,ny-2,z] + ψy[1,ny-1,z])/2.0        
        ψz[0,ny-1,z] = (ψz[0,ny-2,z] + ψz[1,ny-1,z])/2.0

    for i in range(1,nx-1):
        ψx[i,0,0] = (ψx[i,1,0] + ψx[i,0,1])/2.0                            # Front-bottom edge
        ψy[i,0,0] = (ψy[i,1,0] + ψy[i,0,1])/2.0        
        ψz[i,0,0] = (ψz[i,1,0] + ψz[i,0,1])/2.0
        ψx[i,0,nz-1] = (ψx[i,1,nz-1] + ψx[i,0,nz-2])/2.0                   # Back-bottom edge
        ψy[i,0,nz-1] = (ψy[i,1,nz-1] + ψy[i,0,nz-2])/2.0         
        ψz[i,0,nz-1] = (ψz[i,1,nz-1] + ψz[i,0,nz-2])/2.0 
        ψx[i,ny-1,0] = (ψx[i,ny-1,1] + ψx[i,ny-2,0])/2.0                   # Front-top edge
        ψy[i,ny-1,0] = (ψy[i,ny-1,1] + ψy[i,ny-2,0])/2.0         
        ψz[i,ny-1,0] = (ψz[i,ny-1,1] + ψz[i,ny-2,0])/2.0 
        ψx[i,ny-1,nz-1] = (ψx[i,ny-2,nz-1] + ψx[i,ny-1,nz-2])/2.0          # Back-top edge
        ψy[i,ny-1,nz-1] = (ψy[0,ny-2,nz-1] + ψy[1,ny-1,nz-2])/2.0        
        ψz[i,ny-1,nz-1] = (ψz[0,ny-2,nz-1] + ψz[1,ny-1,nz-2])/2.0

    # Vector-potential corner points
    ψx[0,0,0] = (ψx[1,0,0] + ψx[0,1,0] + ψx[0,0,1]) / 3.0                                           # Lower bottom left 
    ψy[0,0,0] = (ψy[1,0,0] + ψy[0,1,0] + ψy[0,0,1]) / 3.0
    ψz[0,0,0] = (ψz[1,0,0] + ψz[0,1,0] + ψz[0,0,1]) / 3.0

    ψx[0,0,nz-1] = (ψx[0,0,nz-2] + ψx[1,0,nz-1] + ψx[0,1,nz-1]) / 3.0                               # Lower back left
    ψy[0,0,nz-1] = (ψy[0,0,nz-2] + ψy[1,0,nz-1] + ψy[0,1,nz-1]) / 3.0
    ψz[0,0,nz-1] = (ψz[0,0,nz-2] + ψz[1,0,nz-1] + ψz[0,1,nz-1]) / 3.0

    ψx[nx-1,0,0] = (ψx[nx-2,0,0] + ψx[nx-1,1,0] + ψx[nx-1,0,1]) / 3.0                               # Lower front right 
    ψy[nx-1,0,0] = (ψy[nx-2,0,0] + ψy[nx-1,1,0] + ψy[nx-1,0,1]) / 3.0
    ψz[nx-1,0,0] = (ψz[nx-2,0,0] + ψz[nx-1,1,0] + ψz[nx-1,0,1]) / 3.0

    ψx[nx-1,0,nz-1] = (ψx[nx-2,0,nz-1] + ψx[nx-1,0,nz-2] + ψx[nx-1,1,nz-1]) / 3.0                   # Lower back right
    ψy[nx-1,0,nz-1] = (ψy[nx-2,0,nz-1] + ψy[nx-1,0,nz-2] + ψy[nx-1,1,nz-1]) / 3.0
    ψz[nx-1,0,nz-1] = (ψz[nx-2,0,nz-1] + ψz[nx-1,0,nz-2] + ψz[nx-1,1,nz-1]) / 3.0

    ψx[0,ny-1,0] = (ψx[1,ny-1,0] + ψx[0,ny-2,0] + ψx[0,ny-1,1]) / 3.0                               # Front top left
    ψy[0,ny-1,0] = (ψy[1,ny-1,0] + ψy[0,ny-2,0] + ψy[0,ny-1,1]) / 3.0
    ψz[0,ny-1,0] = (ψz[1,ny-1,0] + ψz[0,ny-2,0] + ψz[0,ny-1,1]) / 3.0

    ψx[0,ny-1,nz-1] = (ψx[0,ny-1,nz-2] + ψx[1,ny-1,nz-1] + ψx[0,ny-2,nz-1]) / 3.0                   # Back top left
    ψy[0,ny-1,nz-1] = (ψy[0,ny-1,nz-2] + ψy[1,ny-1,nz-1] + ψy[0,ny-2,nz-1]) / 3.0
    ψz[0,ny-1,nz-1] = (ψz[0,ny-1,nz-2] + ψz[1,ny-1,nz-1] + ψz[0,ny-2,nz-1]) / 3.0

    ψx[nx-1,ny-1,0] = (ψx[nx-2,ny-1,0] + ψx[nx-1,ny-2,0] + ψx[nx-1,ny-1,1]) / 3.0                   # Front top right
    ψy[nx-1,ny-1,0] = (ψy[nx-2,ny-1,0] + ψy[nx-1,ny-2,0] + ψy[nx-1,ny-1,1]) / 3.0
    ψz[nx-1,ny-1,0] = (ψz[nx-2,ny-1,0] + ψz[nx-1,ny-2,0] + ψz[nx-1,ny-1,1]) / 3.0

    ψx[nx-1,ny-1,nz-1] = (ψx[nx-2,ny-1,nz-1] + ψx[nx-1,ny-1,nz-2] + ψx[nx-1,ny-2,nz-1]) / 3.0       # Back top right
    ψy[nx-1,ny-1,nz-1] = (ψy[nx-2,ny-1,nz-1] + ψy[nx-1,ny-1,nz-2] + ψy[nx-1,ny-2,nz-1]) / 3.0
    ψz[nx-1,ny-1,nz-1] = (ψz[nx-2,ny-1,nz-1] + ψz[nx-1,ny-1,nz-2] + ψz[nx-1,ny-2,nz-1]) / 3.0



    ψx_sol.append(ψx)
    ψy_sol.append(ψy)
    ψz_sol.append(ψz)



    #---------------------------------------
    # SOLVE FOR THE VELOCITY VECTOR FIELD
    #---------------------------------------
    # using the curl of the vector potential
    # Solve on the internal domain for the velocity field
    for i in range(1,nx-1):
        for j in range(1,ny-1):
            for k in range(1,nz-1):
                u[i,j,k] = (ψz[i,j+1,k] - ψz[i,j-1,k])/dy + (ψy[i,j,k-1] - ψy[i,j,k+1])/dz  # First-order central
                v[i,j,k] = (ψx[i,j,k+1] - ψx[i,j,k-1])/dz + (ψz[i-1,j,k] - ψz[i+1,j,k])/dx
                w[i,j,k] = (ψy[i+1,j,k] - ψy[i-1,j,k])/dx + (ψx[i,j-1,k] - ψx[i,j+1,k])/dy



    # # BOUNDARIES
    # # Apply before or after?
    # for j in range(1,ny-1):
    #     for k in range(1,nz-1):
    #         u[0,j,k] = 0.0                                      # Left wall
    #         v[0,j,k] = -ψz[i+1,j,k]/dx
    #         w[0,j,k] = ψy[i+1,j,k]/dx
    #         u[nx-1,j,k] = ψz[nx-2,j,k]/dx                       # Right wall
    #         v[nx-1,j,k] = -ψy[nx-2,j,k]/dx 
    #         w[nx-1,j,k] = 0.0

    # for i in range(1,nx-1):
    #     for k in range(1,nz-1):
    #         u[i,0,k] = ψz[i,1,k]/dy                             # Bottom wall
    #         v[i,0,k] = 0.0
    #         w[i,j+1,k] = -ψx[i,1,k]/dy
    #         #------------
    #         # TOP WALL
    #         #------------

    # for i in range(1,nx-1):
    #     for j in range(1,ny-1):
    #         u[i,j,0] = -ψy[i,j,1]/dz                            # Front wall
    #         v[i,j,0] = ψx[i,j,1]/dz
    #         w[i,j,0] = 0.0
    #         u[i,j,nz-1] = ψy[i,j,nz-2]/dz                       # Back wall
    #         v[i,j,nz-1] = -ψx[i,j,nz-2]/dz
    #         w[i,j,nz-1] = 0.0

    # OR,

    for j in range(1,ny-1):
        for k in range(1,nz-1):
            u[0,j,k] = 0.0          # Left wall
            v[0,j,k] = 0.0
            w[0,j,k] = 0.0
            u[nx-1,j,k] = 0.0       # Right wall
            v[nx-1,j,k] = 0.0
            w[nx-1,j,k] = 0.0

    for i in range(1,nx-1):
        for j in range(1,ny-1):
            u[i,j,0] = 0.0          # Front wall
            v[i,j,0] = 0.0
            w[i,j,0] = 0.0
            u[i,j,nz-1] = 0.0       # Back wall
            v[i,j,nz-1] = 0.0
            w[i,j,nz-1] = 0.0

    for k in range(1,nz-1):
        for i in range(1,nx-1):
            u[i,0,k] = 0.0          # Bottom wall
            v[i,0,k] = 0.0
            w[i,0,k] = 0.0
            u[i,0,k] = Ut          # Top wall
            v[i,0,k] = 0.0
            w[i,0,k] = 0.0

    u_sol.append(u)
    v_sol.append(v)
    w_sol.append(w)







    #---------------------------------------------------------------------------------
    # SOLVE THE 3D VORTICITY TRANSPORT EQUATION INCLUDING THE VORTEX STRETCHING TERM
    #---------------------------------------------------------------------------------
    # We solve three equations, one for each vorticity component

    #------------------------------
    # Ω_X
    #-------------------------------
    Ωx = Ωxn.copy()
    for i in range(1,nx-1):
        for j in range(1,ny-1):
            for k in range(1,nz-1):
                Cx = u[i,j,k] * (Ωxn[i+1,j,k] - Ωxn[i-1,j,k])/(2*dx)
                Cy = v[i,j,k] * (Ωxn[i,j+1,k] - Ωxn[i,j-1,k])/(2*dy)
                Cz = w[i,j,k] * (Ωxn[i,j,k+1] - Ωxn[i,j,k-1])/(2*dz)

                Dx = (Ωxn[i+1,j,k] - Ωxn[i-1,j,k] + 2*Ωxn[i,j,k])/(dx**2)
                Dy = (Ωxn[i,j+1,k] - Ωxn[i,j-1,k] + 2*Ωxn[i,j,k])/(dy**2)
                Dz = (Ωxn[i,j,k+1] - Ωxn[i,j,k-1] + 2*Ωxn[i,j,k])/(dz**2)

                Ux = Ωxn[i,j,k] * (u[i+1,j,k] - u[i-1,j,k])/(2*dx)
                Uy = Ωyn[i,j,k] * (u[i,j+1,k] - u[i,j-1,k])/(2*dy)
                Uz = Ωzn[i,j,k] * (u[i,j,k+1] - u[i,j,k-1])/(2*dz)

                # The equation
                Ωx[i,j,k] = Ωxn[i,j,k] + dt * (nu * (Dx + Dy + Dz) + Ux + Uy + Uz - (Cx + Cy + Cz))

    #-------------------------------
    # Ω_Y
    #-------------------------------
    Ωy = Ωyn.copy()
    for i in range(1,nx-1):
        for j in range(1,ny-1):
            for k in range(1,nz-1):
                Cx = u[i,j,k] * (Ωyn[i+1,j,k] - Ωyn[i-1,j,k])/(2*dx)
                Cy = v[i,j,k] * (Ωyn[i,j+1,k] - Ωyn[i,j-1,k])/(2*dy)
                Cz = w[i,j,k] * (Ωyn[i,j,k+1] - Ωyn[i,j,k-1])/(2*dz)

                Dx = (Ωyn[i+1,j,k] - Ωyn[i-1,j,k] + 2*Ωyn[i,j,k])/(dx**2)
                Dy = (Ωyn[i,j+1,k] - Ωyn[i,j-1,k] + 2*Ωyn[i,j,k])/(dy**2)
                Dz = (Ωyn[i,j,k+1] - Ωyn[i,j,k-1] + 2*Ωyn[i,j,k])/(dz**2)

                Ux = Ωxn[i,j,k] * (v[i+1,j,k] - v[i-1,j,k])/(2*dx)
                Uy = Ωyn[i,j,k] * (v[i,j+1,k] - v[i,j-1,k])/(2*dy)
                Uz = Ωzn[i,j,k] * (v[i,j,k+1] - v[i,j,k-1])/(2*dz)

                # The equation
                Ωy[i,j,k] = Ωyn[i,j,k] + dt * (nu * (Dx + Dy + Dz) + Ux + Uy + Uz - (Cx + Cy + Cz))

    
    #-------------------------------
    # Ω_Z
    #-------------------------------
    Ωz = Ωzn.copy()
    for i in range(1,nx-1):
        for j in range(1,ny-1):
            for k in range(1,nz-1):
                Cx = u[i,j,k] * (Ωzn[i+1,j,k] - Ωzn[i-1,j,k])/(2*dx)
                Cy = v[i,j,k] * (Ωzn[i,j+1,k] - Ωzn[i,j-1,k])/(2*dy)
                Cz = w[i,j,k] * (Ωzn[i,j,k+1] - Ωzn[i,j,k-1])/(2*dz)

                Dx = (Ωzn[i+1,j,k] - Ωzn[i-1,j,k] + 2*Ωzn[i,j,k])/(dx**2)
                Dy = (Ωzn[i,j+1,k] - Ωzn[i,j-1,k] + 2*Ωzn[i,j,k])/(dy**2)
                Dz = (Ωzn[i,j,k+1] - Ωzn[i,j,k-1] + 2*Ωzn[i,j,k])/(dz**2)

                Ux = Ωxn[i,j,k] * (w[i+1,j,k] - w[i-1,j,k])/(2*dx)
                Uy = Ωyn[i,j,k] * (w[i,j+1,k] - w[i,j-1,k])/(2*dy)
                Uz = Ωzn[i,j,k] * (w[i,j,k+1] - w[i,j,k-1])/(2*dz)

                # The equation
                Ωz[i,j,k] = Ωzn[i,j,k] + dt * (nu * (Dx + Dy + Dz) + Ux + Uy + Uz - (Cx + Cy + Cz))



    # Re-apply the vorticity boundary conditions
    for j in range(1,ny-1):
        for k in range(1,nz-1):
            Ωx[0,j,k] = 0.0                                 # Left wall
            Ωy[0,j,k] = -w[1,j,k]/dx
            Ωz[0,j,k] = v[1,j,k]/dx
            Ωx[nx-1,j,k] = 0.0                              # Right wall
            Ωy[nx-1,j,k] = -w[nx-2,j,k]/dx
            Ωz[nx-1,j,k] =-v[nx-2,j,k]/dx

    for i in range(1,nx-1):
        for j in range(1,ny-1):
            Ωx[i,j,0] = -v[i,j,1]/dz                        # Front wall
            Ωy[i,j,0] = u[i,j,1]/dz
            Ωz[i,j,0] = 0.0
            Ωx[i,j,nz-1] = v[i,j,nz-2]/dz                   # Back wall
            Ωy[i,j,nz-1] = -u[i,j,nz-2]/dz
            Ωz[i,j,nz-1] = 0.0
            
    for i in range(1,nx-1):
        for k in range(1,nz-1):
            Ωx[i,0,k] = w[i,1,k]/dy                         # Bottom wall
            Ωy[i,0,k] = 0.0
            Ωz[i,0,k] = -u[i,1,k]/dy
            Ωx[i,ny-1,k] = -w[i,ny-2,k]/dy                     # Top wall
            Ωy[i,ny-1,k] = (u[i,ny-1,k+1] - u[i,ny-1,k])/dy
            Ωz[i,ny-1,k] = -(u[i,ny-1,k] - u[i,ny-2,k])/dy

    # Vorticity edge points
    for j in range(1,ny-1):
        Ωx[0,j,0] = (Ωx[1,j,0] + Ωx[0,j,1])/2.0                            # Front-left edge
        Ωy[0,j,0] = (Ωy[1,j,0] + Ωy[0,j,1])/2.0
        Ωz[0,j,0] = (Ωz[1,j,0] + Ωz[0,j,1])/2.0
        Ωx[nx-1,j,0] = (Ωx[nx-2,j,0] + Ωx[nx-1,j,1])/2.0                   # Front-right edge
        Ωy[nx-1,j,0] = (Ωy[nx-2,j,0] + Ωy[nx-1,j,1])/2.0
        Ωz[nx-1,j,0] = (Ωz[nx-2,j,0] + Ωz[nx-1,j,1])/2.0
        Ωx[nx-1,j,nz-1] = (Ωx[nx-2,j,nz-1] + Ωx[nx-1,j,nz-2])/2.0          # Back-right edge
        Ωy[nx-1,j,nz-1] = (Ωy[nx-2,j,nz-1] + Ωy[nx-1,j,nz-2])/2.0
        Ωz[nx-1,j,nz-1] = (Ωz[nx-2,j,nz-1] + Ωz[nx-1,j,nz-2])/2.0
        Ωx[0,j,nz-1] = (Ωx[1,j,nz-1] + Ωx[0,j,nz-2])/2.0                   # Back-left edge
        Ωy[0,j,nz-1] = (Ωy[1,j,nz-1] + Ωy[0,j,nz-2])/2.0
        Ωz[0,j,nz-1] = (Ωz[1,j,nz-1] + Ωz[0,j,nz-2])/2.0

    for z in range(1,nz-1):
        Ωx[0,0,z] = (Ωx[1,0,z] + Ωx[0,1,z])/2.0                            # Bottom-left edge
        Ωy[0,0,z] = (Ωy[1,0,z] + Ωy[0,1,z])/2.0        
        Ωz[0,0,z] = (Ωz[1,0,z] + Ωz[0,1,z])/2.0
        Ωx[nx-1,0,z] = (Ωx[nx-2,0,z] + Ωx[0,1,z])/2.0                      # Bottom-right edge
        Ωy[nx-1,0,z] = (Ωy[nx-2,0,z] + Ωy[0,1,z])/2.0        
        Ωz[nx-1,0,z] = (Ωz[nx-2,0,z] + Ωz[0,1,z])/2.0
        Ωx[nx-1,ny-1,z] = (Ωx[nx-2,ny-1,z] + Ωx[nx-1,ny-2,z])/2.0          # Top-right edge
        Ωy[nx-1,ny-1,z] = (Ωy[nx-2,ny-1,z] + Ωy[nx-1,ny-2,z])/2.0        
        Ωz[nx-1,ny-1,z] = (Ωz[nx-2,ny-1,z] + Ωz[nx-1,ny-2,z])/2.0
        Ωx[0,ny-1,z] = (Ωx[0,ny-2,z] + Ωx[1,ny-1,z])/2.0                   # Top-left edge
        Ωy[0,ny-1,z] = (Ωy[0,ny-2,z] + Ωy[1,ny-1,z])/2.0        
        Ωz[0,ny-1,z] = (Ωz[0,ny-2,z] + Ωz[1,ny-1,z])/2.0

    for i in range(1,nx-1):
        Ωx[i,0,0] = (Ωx[i,1,0] + Ωx[i,0,1])/2.0                            # Front-bottom edge
        Ωy[i,0,0] = (Ωy[i,1,0] + Ωy[i,0,1])/2.0        
        Ωz[i,0,0] = (Ωz[i,1,0] + Ωz[i,0,1])/2.0
        Ωx[i,0,nz-1] = (Ωx[i,1,nz-1] + Ωx[i,0,nz-2])/2.0                   # Back-bottom edge
        Ωy[i,0,nz-1] = (Ωy[i,1,nz-1] + Ωy[i,0,nz-2])/2.0         
        Ωz[i,0,nz-1] = (Ωz[i,1,nz-1] + Ωz[i,0,nz-2])/2.0 
        Ωx[i,ny-1,0] = (Ωx[i,ny-1,1] + Ωx[i,ny-2,0])/2.0                   # Front-top edge
        Ωy[i,ny-1,0] = (Ωy[i,ny-1,1] + Ωy[i,ny-2,0])/2.0         
        Ωz[i,ny-1,0] = (Ωz[i,ny-1,1] + Ωz[i,ny-2,0])/2.0 
        Ωx[i,ny-1,nz-1] = (Ωx[i,ny-2,nz-1] + Ωx[i,ny-1,nz-2])/2.0          # Back-top edge
        Ωy[i,ny-1,nz-1] = (Ωy[0,ny-2,nz-1] + Ωy[1,ny-1,nz-2])/2.0        
        Ωz[i,ny-1,nz-1] = (Ωz[0,ny-2,nz-1] + Ωz[1,ny-1,nz-2])/2.0

    # Vorticity corner points
    Ωx[0,0,0] = (Ωx[1,0,0] + Ωx[0,1,0] + Ωx[0,0,1]) / 3.0
    Ωy[0,0,0] = (Ωy[1,0,0] + Ωy[0,1,0] + Ωy[0,0,1]) / 3.0
    Ωz[0,0,0] = (Ωz[1,0,0] + Ωz[0,1,0] + Ωz[0,0,1]) / 3.0

    Ωx[0,0,nz-1] = (Ωx[0,0,nz-2] + Ωx[1,0,nz-1] + Ωx[0,1,nz-1]) / 3.0
    Ωy[0,0,nz-1] = (Ωy[0,0,nz-2] + Ωy[1,0,nz-1] + Ωy[0,1,nz-1]) / 3.0
    Ωz[0,0,nz-1] = (Ωz[0,0,nz-2] + Ωz[1,0,nz-1] + Ωz[0,1,nz-1]) / 3.0

    Ωx[nx-1,0,0] = (Ωx[nx-2,0,0] + Ωx[nx-1,1,0] + Ωx[nx-1,0,1]) / 3.0
    Ωy[nx-1,0,0] = (Ωy[nx-2,0,0] + Ωy[nx-1,1,0] + Ωy[nx-1,0,1]) / 3.0
    ψz[nx-1,0,0] = (ψz[nx-2,0,0] + ψz[nx-1,1,0] + ψz[nx-1,0,1]) / 3.0

    Ωx[nx-1,0,nz-1] = (Ωx[nx-2,0,nz-1] + Ωx[nx-1,0,nz-2] + Ωx[nx-1,1,nz-1]) / 3.0
    Ωy[nx-1,0,nz-1] = (Ωy[nx-2,0,nz-1] + Ωy[nx-1,0,nz-2] + Ωy[nx-1,1,nz-1]) / 3.0
    Ωz[nx-1,0,nz-1] = (Ωz[nx-2,0,nz-1] + Ωz[nx-1,0,nz-2] + Ωz[nx-1,1,nz-1]) / 3.0

    Ωx[0,ny-1,0] = (Ωx[1,ny-1,0] + Ωx[0,ny-2,0] + Ωx[0,ny-1,1]) / 3.0
    Ωy[0,ny-1,0] = (Ωy[1,ny-1,0] + Ωy[0,ny-2,0] + Ωy[0,ny-1,1]) / 3.0
    Ωz[0,ny-1,0] = (Ωz[1,ny-1,0] + Ωz[0,ny-2,0] + Ωz[0,ny-1,1]) / 3.0

    Ωx[0,ny-1,nz-1] = (Ωx[0,ny-1,nz-2] + Ωx[1,ny-1,nz-1] + Ωx[0,ny-2,nz-1]) / 3.0
    Ωy[0,ny-1,nz-1] = (Ωy[0,ny-1,nz-2] + Ωy[1,ny-1,nz-1] + Ωy[0,ny-2,nz-1]) / 3.0
    Ωz[0,ny-1,nz-1] = (Ωz[0,ny-1,nz-2] + Ωz[1,ny-1,nz-1] + Ωz[0,ny-2,nz-1]) / 3.0

    Ωx[nx-1,ny-1,0] = (Ωx[nx-2,ny-1,0] + Ωx[nx-1,ny-2,0] + Ωx[nx-1,ny-1,1]) / 3.0
    Ωy[nx-1,ny-1,0] = (Ωy[nx-2,ny-1,0] + Ωy[nx-1,ny-2,0] + Ωy[nx-1,ny-1,1]) / 3.0
    Ωz[nx-1,ny-1,0] = (Ωz[nx-2,ny-1,0] + Ωz[nx-1,ny-2,0] + Ωz[nx-1,ny-1,1]) / 3.0

    Ωx[nx-1,ny-1,nz-1] = (Ωx[nx-2,ny-1,nz-1] + Ωx[nx-1,ny-1,nz-2] + Ωx[nx-1,ny-2,nz-1]) / 3.0
    Ωy[nx-1,ny-1,nz-1] = (Ωy[nx-2,ny-1,nz-1] + Ωy[nx-1,ny-1,nz-2] + Ωy[nx-1,ny-2,nz-1]) / 3.0
    Ωz[nx-1,ny-1,nz-1] = (Ωz[nx-2,ny-1,nz-1] + Ωz[nx-1,ny-1,nz-2] + Ωz[nx-1,ny-2,nz-1]) / 3.0


    # Store the solution
    Ωx_sol.append(Ωx.copy())
    Ωy_sol.append(Ωy.copy())
    Ωz_sol.append(Ωz.copy())





    t = t + dt
    print(f"t = {t}")


