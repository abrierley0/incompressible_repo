import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sys
import threading
import time
import yaml



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
# 4th August 2025




# Wall time 
elapsed_time = 0

def timer():
    global elapsed_time
    start = time.time()
    while True:
        elapsed_time = time.time() - start
        time.sleep(1)

threading.Thread(target=timer, daemon=True).start()


# Open and read the input parameters
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)


# Grid settings
nx = config['nx']
ny = config['ny']
nz = config['nz']
Lx = config['Lx']
Ly = config['Ly']
Lz = config['Lz']
dx = Lx/(nx-1)
dy = Ly/(ny-1)
dz = Lz/(nz-1)


# Physical parameters
nu = config['nu']
Ut = config['Ut']
Re = Ut*Lx/nu

print()
print('THE VECTOR-POTENTIAL AND VORTICITY FORMULATION FOR THE DRIVEN CUBE')
print()
print('---------------------')
print(f"Re = {Re}")
print(f'Ut = {Ut}')
print(f"nx = {nx}")


# Initialise arrays
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

# Left wall
ψx0[0,1:ny-1,1:nz-1] = ψx0[1,1:ny-1,1:nz-1]
ψy0[0,1:ny-1,1:nz-1] = 0.0
ψz0[0,1:ny-1,1:nz-1] = 0.0

# Right wall
ψx0[nx-1,1:ny-1,1:nz-1] = ψx0[nx-2,1:ny-1,1:nz-1]
ψy0[nx-1,1:ny-1,1:nz-1] = 0.0
ψz0[nx-1,1:ny-1,1:nz-1] = 0.0

# Front wall
ψx0[1:nx-1,1:ny-1,0] = 0.0
ψy0[1:nx-1,1:ny-1,0] = 0.0
ψz0[1:nx-1,1:ny-1,0] = ψz0[1:nx-1,1:ny-1,1]

# Back wall
ψx0[1:nx-1,1:ny-1,nz-1] = 0.0
ψy0[1:nx-1,1:ny-1,nz-1] = 0.0
ψz0[1:nx-1,1:ny-1,nz-1] = ψz0[1:nx-1,1:ny-1,nz-2]

# Bottom wall
ψx0[1:nx-1,0,1:nz-1] = 0.0
ψy0[1:nx-1,0,1:nz-1] = ψy0[1:nx-1,1,1:nz-1]
ψz0[1:nx-1,0,1:nz-1] = 0.0

# Top wall
ψx0[1:nx-1,ny-1,1:nz-1] = 0.0
ψy0[1:nx-1,ny-1,1:nz-1] = ψy0[1:nx-1,ny-2,1:nz-1]
ψz0[1:nx-1,ny-1,1:nz-1] = 0.0


# Vector potential edge points

# y direction edges
ψx0[0,1:ny-1,0] = (ψx0[1,1:ny-1,0] + ψx0[0,1:ny-1,1])/2.0                    # Front left 
ψy0[0,1:ny-1,0] = (ψy0[1,1:ny-1,0] + ψy0[0,1:ny-1,1])/2.0
ψz0[0,1:ny-1,0] = (ψz0[1,1:ny-1,0] + ψz0[0,1:ny-1,1])/2.0
ψx0[nx-1,1:ny-1,0] = (ψx0[nx-2,1:ny-1,0] + ψx0[nx-1,1:ny-1,1])/2.0            # Front right
ψy0[nx-1,1:ny-1,0] = (ψy0[nx-2,1:ny-1,0] + ψy0[nx-1,1:ny-1,1])/2.0
ψz0[nx-1,1:ny-1,0] = (ψz0[nx-2,1:ny-1,0] + ψz0[nx-1,1:ny-1,1])/2.0
ψx0[nx-1,1:ny-1,nz-1] = (ψx0[nx-1,1:ny-1,nz-2] + ψx0[nx-2,1:ny-1,nz-1])/2.0  # Back right
ψy0[nx-1,1:ny-1,nz-1] = (ψy0[nx-1,1:ny-1,nz-2] + ψy0[nx-2,1:ny-1,nz-1])/2.0
ψz0[nx-1,1:ny-1,nz-1] = (ψz0[nx-1,1:ny-1,nz-2] + ψz0[nx-2,1:ny-1,nz-1])/2.0
ψx0[0,1:ny-1,nz-1] = (ψx0[1,1:ny-1,nz-1] + ψx0[0,1:ny-1,nz-2])/2.0           # Back-left edge
ψy0[0,1:ny-1,nz-1] = (ψy0[1,1:ny-1,nz-1] + ψy0[0,1:ny-1,nz-2])/2.0
ψz0[0,1:ny-1,nz-1] = (ψz0[1,1:ny-1,nz-1] + ψz0[0,1:ny-1,nz-2])/2.0

# z direction edges
ψx0[0,0,1:nz-1] = (ψx0[1,0,1:nz-1] + ψx0[0,1,1:nz-1])/2.0                            # Bottom-left edge
ψy0[0,0,1:nz-1] = (ψy0[1,0,1:nz-1] + ψy0[0,1,1:nz-1])/2.0        
ψz0[0,0,1:nz-1] = (ψz0[1,0,1:nz-1] + ψz0[0,1,1:nz-1])/2.0
ψx0[nx-1,0,1:nz-1] = (ψx0[nx-2,0,1:nz-1] + ψx0[nx-1,1,1:nz-1])/2.0                   # Bottom-right edge
ψy0[nx-1,0,1:nz-1] = (ψy0[nx-2,0,1:nz-1] + ψy0[nx-1,1,1:nz-1])/2.0        
ψz0[nx-1,0,1:nz-1] = (ψz0[nx-2,0,1:nz-1] + ψz0[nx-1,1,1:nz-1])/2.0
ψx0[nx-1,ny-1,1:nz-1] = (ψx0[nx-2,ny-1,1:nz-1] + ψx0[nx-1,ny-2,1:nz-1])/2.0          # Top-right edge
ψy0[nx-1,ny-1,1:nz-1] = (ψy0[nx-2,ny-1,1:nz-1] + ψy0[nx-1,ny-2,1:nz-1])/2.0        
ψz0[nx-1,ny-1,1:nz-1] = (ψz0[nx-2,ny-1,1:nz-1] + ψz0[nx-1,ny-2,1:nz-1])/2.0
ψx0[0,ny-1,1:nz-1] = (ψx0[0,ny-2,1:nz-1] + ψx0[1,ny-1,1:nz-1])/2.0                   # Top-left edge
ψy0[0,ny-1,1:nz-1] = (ψy0[0,ny-2,1:nz-1] + ψy0[1,ny-1,1:nz-1])/2.0        
ψz0[0,ny-1,1:nz-1] = (ψz0[0,ny-2,1:nz-1] + ψz0[1,ny-1,1:nz-1])/2.0

# x direction edges
ψx0[1:nx-1,0,0] = (ψx0[1:nx-1,1,0] + ψx0[1:nx-1,0,1])/2.0                            # Front-bottom edge
ψy0[1:nx-1,0,0] = (ψy0[1:nx-1,1,0] + ψy0[1:nx-1,0,1])/2.0        
ψz0[1:nx-1,0,0] = (ψz0[1:nx-1,1,0] + ψz0[1:nx-1,0,1])/2.0
ψx0[1:nx-1,0,nz-1] = (ψx0[1:nx-1,1,nz-1] + ψx0[1:nx-1,0,nz-2])/2.0                   # Back-bottom edge
ψy0[1:nx-1,0,nz-1] = (ψy0[1:nx-1,1,nz-1] + ψy0[1:nx-1,0,nz-2])/2.0         
ψz0[1:nx-1,0,nz-1] = (ψz0[1:nx-1,1,nz-1] + ψz0[1:nx-1,0,nz-2])/2.0 
ψx0[1:nx-1,ny-1,0] = (ψx0[1:nx-1,ny-1,1] + ψx0[1:nx-1,ny-2,0])/2.0                   # Front-top edge
ψy0[1:nx-1,ny-1,0] = (ψy0[1:nx-1,ny-1,1] + ψy0[1:nx-1,ny-2,0])/2.0         
ψz0[1:nx-1,ny-1,0] = (ψz0[1:nx-1,ny-1,1] + ψz0[1:nx-1,ny-2,0])/2.0 
ψx0[1:nx-1,ny-1,nz-1] = (ψx0[1:nx-1,ny-2,nz-1] + ψx0[1:nx-1,ny-1,nz-2])/2.0          # Back-top edge
ψy0[1:nx-1,ny-1,nz-1] = (ψy0[1:nx-1,ny-2,nz-1] + ψy0[1:nx-1,ny-1,nz-2])/2.0        
ψz0[1:nx-1,ny-1,nz-1] = (ψz0[1:nx-1,ny-2,nz-1] + ψz0[1:nx-1,ny-1,nz-2])/2.0

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

np.set_printoptions(linewidth=1000, threshold=np.inf, precision=1, suppress=True)  # Ensure full matrix prints
#np.set_printoptions(linewidth=1000, threshold=np.inf)

# print()
# print(f"ψx0 is : ")
# print()
# print(ψx0[:,:,3])
# print()


#--------------------------------------------
# VELOCITY VECTOR FIELD BOUNDARY CONDITIONS
#--------------------------------------------

# MAIN VELOCITY BOUNDARY CONDITIONS

u0[0,1:ny-1,1:nz-1] = 0.0          # Left wall
v0[0,1:ny-1,1:nz-1] = 0.0
w0[0,1:ny-1,1:nz-1] = 0.0
u0[nx-1,1:ny-1,1:nz-1] = 0.0       # Right wall
v0[nx-1,1:ny-1,1:nz-1] = 0.0
w0[nx-1,1:ny-1,1:nz-1] = 0.0


u0[1:nx-1,1:ny-1,0] = 0.0          # Front wall
v0[1:nx-1,1:ny-1,0] = 0.0
w0[1:nx-1,1:ny-1,0] = 0.0
u0[1:nx-1,1:ny-1,nz-1] = 0.0       # Back wall
v0[1:nx-1,1:ny-1,nz-1] = 0.0
w0[1:nx-1,1:ny-1,nz-1] = 0.0


u0[1:nx-1,0,1:nz-1] = 0.0          # Bottom wall
v0[1:nx-1,0,1:nz-1] = 0.0
w0[1:nx-1,0,1:nz-1] = 0.0
u0[1:nx-1,ny-1,1:nz-1] = Ut        # Top wall
v0[1:nx-1,ny-1,1:nz-1] = 0.0
w0[1:nx-1,ny-1,1:nz-1] = 0.0

# Velocity edge points


u0[0,1:ny-1,0] = (u0[1,1:ny-1,0] + u0[0,1:ny-1,1])/2.0                            # Front-left edge
v0[0,1:ny-1,0] = (v0[1,1:ny-1,0] + v0[0,1:ny-1,1])/2.0
w0[0,1:ny-1,0] = (w0[1,1:ny-1,0] + w0[0,1:ny-1,1])/2.0
u0[nx-1,1:ny-1,0] = (u0[nx-2,1:ny-1,0] + u0[nx-1,1:ny-1,1])/2.0                   # Front-right edge
v0[nx-1,1:ny-1,0] = (v0[nx-2,1:ny-1,0] + v0[nx-1,1:ny-1,1])/2.0
w0[nx-1,1:ny-1,0] = (w0[nx-2,1:ny-1,0] + w0[nx-1,1:ny-1,1])/2.0
u0[nx-1,1:ny-1,nz-1] = (u0[nx-2,1:ny-1,nz-1] + u0[nx-1,1:ny-1,nz-2])/2.0          # Back-right edge
v0[nx-1,1:ny-1,nz-1] = (v0[nx-2,1:ny-1,nz-1] + v0[nx-1,1:ny-1,nz-2])/2.0
w0[nx-1,1:ny-1,nz-1] = (w0[nx-2,1:ny-1,nz-1] + w0[nx-1,1:ny-1,nz-2])/2.0
u0[0,1:ny-1,nz-1] = (u0[1,1:ny-1,nz-1] + u0[0,1:ny-1,nz-2])/2.0                   # Back-left edge
v0[0,1:ny-1,nz-1] = (v0[1,1:ny-1,nz-1] + v0[0,1:ny-1,nz-2])/2.0
w0[0,1:ny-1,nz-1] = (w0[1,1:ny-1,nz-1] + w0[0,1:ny-1,nz-2])/2.0


u0[0,0,1:nz-1] = (u0[1,0,1:nz-1] + u0[0,1,1:nz-1])/2.0                            # Bottom-left edge
v0[0,0,1:nz-1] = (v0[1,0,1:nz-1] + v0[0,1,1:nz-1])/2.0        
w0[0,0,1:nz-1] = (w0[1,0,1:nz-1] + w0[0,1,1:nz-1])/2.0
u0[nx-1,0,1:nz-1] = (u0[nx-2,0,1:nz-1] + u0[nx-1,1,1:nz-1])/2.0                      # Bottom-right edge
v0[nx-1,0,1:nz-1] = (v0[nx-2,0,1:nz-1] + v0[nx-1,1,1:nz-1])/2.0        
w0[nx-1,0,1:nz-1] = (w0[nx-2,0,1:nz-1] + w0[nx-1,1,1:nz-1])/2.0
u0[nx-1,ny-1,1:nz-1] = (u0[nx-2,ny-1,1:nz-1] + u0[nx-1,ny-2,1:nz-1])/2.0          # Top-right edge
v0[nx-1,ny-1,1:nz-1] = (v0[nx-2,ny-1,1:nz-1] + v0[nx-1,ny-2,1:nz-1])/2.0        
w0[nx-1,ny-1,1:nz-1] = (w0[nx-2,ny-1,1:nz-1] + w0[nx-1,ny-2,1:nz-1])/2.0
u0[0,ny-1,1:nz-1] = (u0[0,ny-2,1:nz-1] + u0[1,ny-1,1:nz-1])/2.0                   # Top-left edge
v0[0,ny-1,1:nz-1] = (v0[0,ny-2,1:nz-1] + v0[1,ny-1,1:nz-1])/2.0        
w0[0,ny-1,1:nz-1] = (w0[0,ny-2,1:nz-1] + w0[1,ny-1,1:nz-1])/2.0


u0[1:nx-1,0,0] = (u0[1:nx-1,1,0] + u0[1:nx-1,0,1])/2.0                            # Front-bottom edge
v0[1:nx-1,0,0] = (v0[1:nx-1,1,0] + v0[1:nx-1,0,1])/2.0        
w0[1:nx-1,0,0] = (w0[1:nx-1,1,0] + w0[1:nx-1,0,1])/2.0
u0[1:nx-1,0,nz-1] = (u0[1:nx-1,1,nz-1] + u0[1:nx-1,0,nz-2])/2.0                   # Back-bottom edge
v0[1:nx-1,0,nz-1] = (v0[1:nx-1,1,nz-1] + v0[1:nx-1,0,nz-2])/2.0         
w0[1:nx-1,0,nz-1] = (w0[1:nx-1,1,nz-1] + w0[1:nx-1,0,nz-2])/2.0 
u0[1:nx-1,ny-1,0] = (u0[1:nx-1,ny-1,1] + u0[1:nx-1,ny-2,0])/2.0                   # Front-top edge
v0[1:nx-1,ny-1,0] = (v0[1:nx-1,ny-1,1] + v0[1:nx-1,ny-2,0])/2.0         
w0[1:nx-1,ny-1,0] = (w0[1:nx-1,ny-1,1] + w0[1:nx-1,ny-2,0])/2.0 
u0[1:nx-1,ny-1,nz-1] = (u0[1:nx-1,ny-2,nz-1] + u0[1:nx-1,ny-1,nz-2])/2.0          # Back-top edge
v0[1:nx-1,ny-1,nz-1] = (v0[1:nx-1,ny-2,nz-1] + v0[1:nx-1,ny-1,nz-2])/2.0        
w0[1:nx-1,ny-1,nz-1] = (w0[1:nx-1,ny-2,nz-1] + w0[1:nx-1,ny-1,nz-2])/2.0

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

# print()
# print(f"u0 is : ")
# print()
# print(u0[:,:,3])
# print()

#------------------------------------------------
# VORTICITY VECTOR FIELD (Ω) BOUNDARY CONDITIONS
#------------------------------------------------
# derived using the definition of vorticity

Ωx0[0,1:ny-1,1:nz-1] = 0.0                                        # Left wall
Ωy0[0,1:ny-1,1:nz-1] = -w0[1,1:ny-1,1:nz-1]/dx
Ωz0[0,1:ny-1,1:nz-1] = v0[1,1:ny-1,1:nz-1]/dx
Ωx0[nx-1,1:ny-1,1:nz-1] = 0.0                                     # Right wall
Ωy0[nx-1,1:ny-1,1:nz-1] = w0[nx-2,1:ny-1,1:nz-1]/dx
Ωz0[nx-1,1:ny-1,1:nz-1] =-v0[nx-2,1:ny-1,1:nz-1]/dx


Ωx0[1:nx-1,1:ny-1,0] = -v0[1:nx-1,1:ny-1,1]/dz                              # Front wall
Ωy0[1:nx-1,1:ny-1,0] = u0[1:nx-1,1:ny-1,1]/dz
Ωz0[1:nx-1,1:ny-1,0] = 0.0
Ωx0[1:nx-1,1:ny-1,nz-1] = v0[1:nx-1,1:ny-1,nz-2]/dz                         # Back wall
Ωy0[1:nx-1,1:ny-1,nz-1] = -u0[1:nx-1,1:ny-1,nz-2]/dz
Ωz0[1:nx-1,1:ny-1,nz-1] = 0.0
        

Ωx0[1:nx-1,0,1:nz-1] = w0[1:nx-1,1,1:nz-1]/dy                               # Bottom wall
Ωy0[1:nx-1,0,1:nz-1] = 0.0
Ωz0[1:nx-1,0,1:nz-1] = -u0[1:nx-1,1,1:nz-1]/dy
Ωx0[1:nx-1,ny-1,1:nz-1] = -w0[1:nx-1,ny-2,1:nz-1]/dy                        # Top wall
Ωy0[1:nx-1,ny-1,1:nz-1] = 0.0                                     
Ωz0[1:nx-1,ny-1,1:nz-1] = -(Ut - u0[1:nx-1,ny-2,1:nz-1])/dy

# Vorticity edge points

Ωx0[0,1:ny-1,0] = (Ωx0[1,1:ny-1,0] + Ωx0[0,1:ny-1,1])/2.0                            # Front-left edge
Ωy0[0,1:ny-1,0] = (Ωy0[1,1:ny-1,0] + Ωy0[0,1:ny-1,1])/2.0
Ωz0[0,1:ny-1,0] = (Ωz0[1,1:ny-1,0] + Ωz0[0,1:ny-1,1])/2.0
Ωx0[nx-1,1:ny-1,0] = (Ωx0[nx-2,1:ny-1,0] + Ωx0[nx-1,1:ny-1,1])/2.0                   # Front-right edge
Ωy0[nx-1,1:ny-1,0] = (Ωy0[nx-2,1:ny-1,0] + Ωy0[nx-1,1:ny-1,1])/2.0
Ωz0[nx-1,1:ny-1,0] = (Ωz0[nx-2,1:ny-1,0] + Ωz0[nx-1,1:ny-1,1])/2.0
Ωx0[nx-1,1:ny-1,nz-1] = (Ωx0[nx-2,1:ny-1,nz-1] + Ωx0[nx-1,1:ny-1,nz-2])/2.0          # Back-right edge
Ωy0[nx-1,1:ny-1,nz-1] = (Ωy0[nx-2,1:ny-1,nz-1] + Ωy0[nx-1,1:ny-1,nz-2])/2.0
Ωz0[nx-1,1:ny-1,nz-1] = (Ωz0[nx-2,1:ny-1,nz-1] + Ωz0[nx-1,1:ny-1,nz-2])/2.0
Ωx0[0,1:ny-1,nz-1] = (Ωx0[1,1:ny-1,nz-1] + Ωx0[0,1:ny-1,nz-2])/2.0                   # Back-left edge
Ωy0[0,1:ny-1,nz-1] = (Ωy0[1,1:ny-1,nz-1] + Ωy0[0,1:ny-1,nz-2])/2.0
Ωz0[0,1:ny-1,nz-1] = (Ωz0[1,1:ny-1,nz-1] + Ωz0[0,1:ny-1,nz-2])/2.0


Ωx0[0,0,1:nz-1] = (Ωx0[1,0,1:nz-1] + Ωx0[0,1,1:nz-1])/2.0                            # Bottom-left edge
Ωy0[0,0,1:nz-1] = (Ωy0[1,0,1:nz-1] + Ωy0[0,1,1:nz-1])/2.0        
Ωz0[0,0,1:nz-1] = (Ωz0[1,0,1:nz-1] + Ωz0[0,1,1:nz-1])/2.0
Ωx0[nx-1,0,1:nz-1] = (Ωx0[nx-2,0,1:nz-1] + Ωx0[nx-1,1,1:nz-1])/2.0                   # Bottom-right edge
Ωy0[nx-1,0,1:nz-1] = (Ωy0[nx-2,0,1:nz-1] + Ωy0[nx-1,1,1:nz-1])/2.0        
Ωz0[nx-1,0,1:nz-1] = (Ωz0[nx-2,0,1:nz-1] + Ωz0[nx-1,1,1:nz-1])/2.0
Ωx0[nx-1,ny-1,1:nz-1] = (Ωx0[nx-2,ny-1,1:nz-1] + Ωx0[nx-1,ny-2,1:nz-1])/2.0          # Top-right edge
Ωy0[nx-1,ny-1,1:nz-1] = (Ωy0[nx-2,ny-1,1:nz-1] + Ωy0[nx-1,ny-2,1:nz-1])/2.0        
Ωz0[nx-1,ny-1,1:nz-1] = (Ωz0[nx-2,ny-1,1:nz-1] + Ωz0[nx-1,ny-2,1:nz-1])/2.0
Ωx0[0,ny-1,1:nz-1] = (Ωx0[0,ny-2,1:nz-1] + Ωx0[1,ny-1,1:nz-1])/2.0                   # Top-left edge
Ωy0[0,ny-1,1:nz-1] = (Ωy0[0,ny-2,1:nz-1] + Ωy0[1,ny-1,1:nz-1])/2.0        
Ωz0[0,ny-1,1:nz-1] = (Ωz0[0,ny-2,1:nz-1] + Ωz0[1,ny-1,1:nz-1])/2.0


Ωx0[1:nx-1,0,0] = (Ωx0[1:nx-1,1,0] + Ωx0[1:nx-1,0,1])/2.0                            # Front-bottom edge
Ωy0[1:nx-1,0,0] = (Ωy0[1:nx-1,1,0] + Ωy0[1:nx-1,0,1])/2.0        
Ωz0[1:nx-1,0,0] = (Ωz0[1:nx-1,1,0] + Ωz0[1:nx-1,0,1])/2.0
Ωx0[1:nx-1,0,nz-1] = (Ωx0[1:nx-1,1,nz-1] + Ωx0[1:nx-1,0,nz-2])/2.0                   # Back-bottom edge
Ωy0[1:nx-1,0,nz-1] = (Ωy0[1:nx-1,1,nz-1] + Ωy0[1:nx-1,0,nz-2])/2.0         
Ωz0[1:nx-1,0,nz-1] = (Ωz0[1:nx-1,1,nz-1] + Ωz0[1:nx-1,0,nz-2])/2.0 
Ωx0[1:nx-1,ny-1,0] = (Ωx0[1:nx-1,ny-1,1] + Ωx0[1:nx-1,ny-2,0])/2.0                   # Front-top edge
Ωy0[1:nx-1,ny-1,0] = (Ωy0[1:nx-1,ny-1,1] + Ωy0[1:nx-1,ny-2,0])/2.0         
Ωz0[1:nx-1,ny-1,0] = (Ωz0[1:nx-1,ny-1,1] + Ωz0[1:nx-1,ny-2,0])/2.0 
Ωx0[1:nx-1,ny-1,nz-1] = (Ωx0[1:nx-1,ny-2,nz-1] + Ωx0[1:nx-1,ny-1,nz-2])/2.0          # Back-top edge
Ωy0[1:nx-1,ny-1,nz-1] = (Ωy0[1:nx-1,ny-2,nz-1] + Ωy0[1:nx-1,ny-1,nz-2])/2.0        
Ωz0[1:nx-1,ny-1,nz-1] = (Ωz0[1:nx-1,ny-2,nz-1] + Ωz0[1:nx-1,ny-1,nz-2])/2.0

# Vorticity corner points
Ωx0[0,0,0] = (Ωx0[1,0,0] + Ωx0[0,1,0] + Ωx0[0,0,1]) / 3.0                                       # Front bottom left 
Ωy0[0,0,0] = (Ωy0[1,0,0] + Ωy0[0,1,0] + Ωy0[0,0,1]) / 3.0
Ωz0[0,0,0] = (Ωz0[1,0,0] + Ωz0[0,1,0] + Ωz0[0,0,1]) / 3.0

Ωx0[0,0,nz-1] = (Ωx0[0,0,nz-2] + Ωx0[1,0,nz-1] + Ωx0[0,1,nz-1]) / 3.0                           # Back bottom left
Ωy0[0,0,nz-1] = (Ωy0[0,0,nz-2] + Ωy0[1,0,nz-1] + Ωy0[0,1,nz-1]) / 3.0
Ωz0[0,0,nz-1] = (Ωz0[0,0,nz-2] + Ωz0[1,0,nz-1] + Ωz0[0,1,nz-1]) / 3.0

Ωx0[nx-1,0,0] = (Ωx0[nx-2,0,0] + Ωx0[nx-1,1,0] + Ωx0[nx-1,0,1]) / 3.0                           # Front bottom right
Ωy0[nx-1,0,0] = (Ωy0[nx-2,0,0] + Ωy0[nx-1,1,0] + Ωy0[nx-1,0,1]) / 3.0
Ωz0[nx-1,0,0] = (Ωz0[nx-2,0,0] + Ωz0[nx-1,1,0] + Ωz0[nx-1,0,1]) / 3.0

Ωx0[nx-1,0,nz-1] = (Ωx0[nx-2,0,nz-1] + Ωx0[nx-1,0,nz-2] + Ωx0[nx-1,1,nz-1]) / 3.0               # Back bottom right
Ωy0[nx-1,0,nz-1] = (Ωy0[nx-2,0,nz-1] + Ωy0[nx-1,0,nz-2] + Ωy0[nx-1,1,nz-1]) / 3.0
Ωz0[nx-1,0,nz-1] = (Ωz0[nx-2,0,nz-1] + Ωz0[nx-1,0,nz-2] + Ωz0[nx-1,1,nz-1]) / 3.0

Ωx0[0,ny-1,0] = (Ωx0[1,ny-1,0] + Ωx0[0,ny-2,0] + Ωx0[0,ny-1,1]) / 3.0                           # Front top left
Ωy0[0,ny-1,0] = (Ωy0[1,ny-1,0] + Ωy0[0,ny-2,0] + Ωy0[0,ny-1,1]) / 3.0
Ωz0[0,ny-1,0] = (Ωz0[1,ny-1,0] + Ωz0[0,ny-2,0] + Ωz0[0,ny-1,1]) / 3.0

Ωx0[0,ny-1,nz-1] = (Ωx0[0,ny-1,nz-2] + Ωx0[1,ny-1,nz-1] + Ωx0[0,ny-2,nz-1]) / 3.0               # Back top left
Ωy0[0,ny-1,nz-1] = (Ωy0[0,ny-1,nz-2] + Ωy0[1,ny-1,nz-1] + Ωy0[0,ny-2,nz-1]) / 3.0
Ωz0[0,ny-1,nz-1] = (Ωz0[0,ny-1,nz-2] + Ωz0[1,ny-1,nz-1] + Ωz0[0,ny-2,nz-1]) / 3.0

Ωx0[nx-1,ny-1,0] = (Ωx0[nx-2,ny-1,0] + Ωx0[nx-1,ny-2,0] + Ωx0[nx-1,ny-1,1]) / 3.0               # Front top right
Ωy0[nx-1,ny-1,0] = (Ωy0[nx-2,ny-1,0] + Ωy0[nx-1,ny-2,0] + Ωy0[nx-1,ny-1,1]) / 3.0
Ωz0[nx-1,ny-1,0] = (Ωz0[nx-2,ny-1,0] + Ωz0[nx-1,ny-2,0] + Ωz0[nx-1,ny-1,1]) / 3.0

Ωx0[nx-1,ny-1,nz-1] = (Ωx0[nx-2,ny-1,nz-1] + Ωx0[nx-1,ny-1,nz-2] + Ωx0[nx-1,ny-2,nz-1]) / 3.0   # Back top right
Ωy0[nx-1,ny-1,nz-1] = (Ωy0[nx-2,ny-1,nz-1] + Ωy0[nx-1,ny-1,nz-2] + Ωy0[nx-1,ny-2,nz-1]) / 3.0
Ωz0[nx-1,ny-1,nz-1] = (Ωz0[nx-2,ny-1,nz-1] + Ωz0[nx-1,ny-1,nz-2] + Ωz0[nx-1,ny-2,nz-1]) / 3.0

# print()
# print(f"Ωz0 is : ")
# print()
# print(Ωz0[:,:,3])
# print()


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


# Poisson-solver parameters
tol = config['tol']
errx = 1e5
erry = 1e5
errz = 1e5
itmax = config['tol']
β = config['β']

# Time marching parameters
dt = config['dt']
#dt = min(0.15 * dx**2 / nu, 4 * nu / (Ut**2))
Ω_conv = config['Ω_conv']

print(f'dt = {dt}')
print(f'Ω_conv = {Ω_conv}')
print('---------------------')
print('Poisson Parameters:')
print(f"β = {β}")
print(f'itmax = {itmax}')
print(f"tol = {tol}")
print('---------------------')
print()

# Start main time loop

t = 0
its = 0
u = u0.copy()
v = v0.copy()
w = w0.copy()
vort_conv = 10  # allow the flow to develop
while vort_conv > Ω_conv:

    #------------------------------------------------------------------
    # SOLVE THREE VECTOR-POTENTIAL POISSON EQUATIONS USING ITERATION
    #------------------------------------------------------------------

    # POISSON SOLVER FOR ψ_x
    it = 0
    errx = 1e5
    ψx = ψx_sol[-1].copy()
    Ωxn = Ωx_sol[-1].copy()
    while it < itmax and errx > tol:
        ψx_k = ψx.copy()
        ψx[1:nx-1,1:ny-1,1:nz-1] = (β / (2*(dx**2*dz**2 + dy**2*dz**2 + dx**2*dy**2))) * (dx**2*dy**2*dz**2*Ωxn[1:nx-1,1:ny-1,1:nz-1] + dy**2*dz**2*(ψx_k[2:nx,1:ny-1,1:nz-1]+ψx_k[0,1:ny-1,1:nz-1]) + dx**2*dz**2*(ψx_k[1:nx-1,2:ny,1:nz-1]+ψx_k[1:nx-1,0:ny-2,1:nz-1]) + dx**2*dy**2*(ψx_k[1:nx-1,1:ny-1,2:nz] + ψx_k[1:nx-1,1:ny-1,0:nz-2])) + (1 - β) * ψx_k[1:nx-1,1:ny-1,1:nz-1]
        errx = np.linalg.norm(ψx.ravel() - ψx_k.ravel())
        it = it + 1
        if it % 50 == 0: 
            print()
            print(f"X Iteration: {it}")
            print(f"X Error: {errx}")

    # POISSON SOLVER FOR ψ_y
    it = 0
    erry = 1e5
    ψy = ψy_sol[-1].copy()
    Ωyn = Ωy_sol[-1].copy()
    while it < itmax and erry > tol:
        ψy_k = ψy.copy()
        ψy[1:nx-1,1:ny-1,1:nz-1] = (β / (2*(dx**2*dz**2 + dy**2*dz**2 + dx**2*dy**2))) * (dx**2*dy**2*dz**2*Ωyn[1:nx-1,1:ny-1,1:nz-1] + dy**2*dz**2*(ψy_k[2:nx,1:ny-1,1:nz-1]+ψy_k[0:nx-2,1:ny-1,1:nz-1]) + dx**2*dz**2*(ψy_k[1:nx-1,2:ny,1:nz-1]+ψy_k[1:nx-1,0:ny-2,1:nz-1]) + dx**2*dy**2*(ψy_k[1:nx-1,1:ny-1,2:nz] + ψy_k[1:nx-1,1:ny-1,0:nz-2])) + (1 - β) * ψy_k[1:nx-1,1:ny-1,1:nz-1]
        erry = np.linalg.norm(ψy.ravel() - ψy_k.ravel())
        it = it + 1
        if it % 50 == 0: 
            print()
            print(f"Y Iteration: {it}")
            print(f"Y Error: {erry}")


    # POISSON SOLVER FOR ψ_z
    it = 0 
    errz = 1e5
    ψz = ψz_sol[-1].copy()
    Ωzn = Ωz_sol[-1].copy()
    #print(f"Ωzn is {Ωzn}")
    while it < itmax and errz > tol:
        ψz_k = ψz.copy()
        ψz[1:nx-1,1:ny-1,1:nz-1] = (β / (2*(dx**2*dz**2 + dy**2*dz**2 + dx**2*dy**2))) * (dx**2*dy**2*dz**2*Ωzn[1:nx-1,1:ny-1,1:nz-1] + dy**2*dz**2*(ψz_k[2:nx,1:ny-1,1:nz-1]+ψz_k[0:nx-2,1:ny-1,1:nz-1]) + dx**2*dz**2*(ψz_k[1:nx-1,2:ny,1:nz-1]+ψz_k[1:nx-1,0:ny-2,1:nz-1]) + dx**2*dy**2*(ψz_k[1:nx-1,1:ny-1,2:nz] + ψz_k[1:nx-1,1:ny-1,0:nz-2])) + (1 - β) * ψz_k[1:nx-1,1:ny-1,1:nz-1]
        errz = np.linalg.norm(ψz.ravel() - ψz_k.ravel())
        it = it + 1
        if it % 50 == 0: 
            print()
            print(f"Z Iteration: {it}")
            print(f"Z Error: {errz:.3f}")

    # print()
    # print(f"ψz at {t:.3f}s pre-BCs :")
    # print()
    # print(ψz[:,:,3])

    # print()
    # print(f"ψy at {t:.3f}s pre-BCs :")
    # print()
    # print(ψy[:,:,3])

    # print()
    # print(f"ψz at {t:.3f}s pre-BCs :")
    # print()
    # print(ψz[:,:,3])


    # #-------------------------------------------
    # # Re-apply boundary conditions to ψ
    # #-------------------------------------------
    ψx[0,1:ny-1,1:nz-1] = ψx[1,1:ny-1,1:nz-1]
    ψy[0,1:ny-1,1:nz-1] = 0.0
    ψz[0,1:ny-1,1:nz-1] = 0.0

    # Right wall
    ψx[nx-1,1:ny-1,1:nz-1] = ψx[nx-2,1:ny-1,1:nz-1]
    ψy[nx-1,1:ny-1,1:nz-1] = 0.0
    ψz[nx-1,1:ny-1,1:nz-1] = 0.0

    # Front wall
    ψx[1:nx-1,1:ny-1,0] = 0.0
    ψy[1:nx-1,1:ny-1,0] = 0.0
    ψz[1:nx-1,1:ny-1,0] = ψz[1:nx-1,1:ny-1,1]

    # Back wall
    ψx[1:nx-1,1:ny-1,nz-1] = 0.0
    ψy[1:nx-1,1:ny-1,nz-1] = 0.0
    ψz[1:nx-1,1:ny-1,nz-1] = ψz[1:nx-1,1:ny-1,nz-2]

    # Bottom wall
    ψx[1:nx-1,0,1:nz-1] = 0.0
    ψy[1:nx-1,0,1:nz-1] = ψy[1:nx-1,1,1:nz-1]
    ψz[1:nx-1,0,1:nz-1] = 0.0

    # Top wall
    ψx[1:nx-1,ny-1,1:nz-1] = 0.0
    ψy[1:nx-1,ny-1,1:nz-1] = ψy[1:nx-1,ny-2,1:nz-1]
    ψz[1:nx-1,ny-1,1:nz-1] = 0.0


    # Vector potential edge points

    # y direction edges
    ψx[0,1:ny-1,0] = (ψx[1,1:ny-1,0] + ψx[0,1:ny-1,1])/2.0                    # Front left 
    ψy[0,1:ny-1,0] = (ψy[1,1:ny-1,0] + ψy[0,1:ny-1,1])/2.0
    ψz[0,1:ny-1,0] = (ψz[1,1:ny-1,0] + ψz[0,1:ny-1,1])/2.0
    ψx[nx-1,1:ny-1,0] = (ψx[nx-2,1:ny-1,0] + ψx[nx-1,1:ny-1,1])/2.0            # Front right
    ψy[nx-1,1:ny-1,0] = (ψy[nx-2,1:ny-1,0] + ψy[nx-1,1:ny-1,1])/2.0
    ψz[nx-1,1:ny-1,0] = (ψz[nx-2,1:ny-1,0] + ψz[nx-1,1:ny-1,1])/2.0
    ψx[nx-1,1:ny-1,nz-1] = (ψx[nx-1,1:ny-1,nz-2] + ψx[nx-2,1:ny-1,nz-1])/2.0  # Back right
    ψy[nx-1,1:ny-1,nz-1] = (ψy[nx-1,1:ny-1,nz-2] + ψy[nx-2,1:ny-1,nz-1])/2.0
    ψz[nx-1,1:ny-1,nz-1] = (ψz[nx-1,1:ny-1,nz-2] + ψz[nx-2,1:ny-1,nz-1])/2.0
    ψx[0,1:ny-1,nz-1] = (ψx[1,1:ny-1,nz-1] + ψx[0,1:ny-1,nz-2])/2.0           # Back-left edge
    ψy[0,1:ny-1,nz-1] = (ψy[1,1:ny-1,nz-1] + ψy[0,1:ny-1,nz-2])/2.0
    ψz[0,1:ny-1,nz-1] = (ψz[1,1:ny-1,nz-1] + ψz[0,1:ny-1,nz-2])/2.0

    # z direction edges
    ψx[0,0,1:nz-1] = (ψx[1,0,1:nz-1] + ψx[0,1,1:nz-1])/2.0                            # Bottom-left edge
    ψy[0,0,1:nz-1] = (ψy[1,0,1:nz-1] + ψy[0,1,1:nz-1])/2.0        
    ψz[0,0,1:nz-1] = (ψz[1,0,1:nz-1] + ψz[0,1,1:nz-1])/2.0
    ψx[nx-1,0,1:nz-1] = (ψx[nx-2,0,1:nz-1] + ψx[nx-1,1,1:nz-1])/2.0                   # Bottom-right edge
    ψy[nx-1,0,1:nz-1] = (ψy[nx-2,0,1:nz-1] + ψy[nx-1,1,1:nz-1])/2.0        
    ψz[nx-1,0,1:nz-1] = (ψz[nx-2,0,1:nz-1] + ψz[nx-1,1,1:nz-1])/2.0
    ψx[nx-1,ny-1,1:nz-1] = (ψx[nx-2,ny-1,1:nz-1] + ψx[nx-1,ny-2,1:nz-1])/2.0          # Top-right edge
    ψy[nx-1,ny-1,1:nz-1] = (ψy[nx-2,ny-1,1:nz-1] + ψy[nx-1,ny-2,1:nz-1])/2.0        
    ψz[nx-1,ny-1,1:nz-1] = (ψz[nx-2,ny-1,1:nz-1] + ψz[nx-1,ny-2,1:nz-1])/2.0
    ψx[0,ny-1,1:nz-1] = (ψx[0,ny-2,1:nz-1] + ψx[1,ny-1,1:nz-1])/2.0                   # Top-left edge
    ψy[0,ny-1,1:nz-1] = (ψy[0,ny-2,1:nz-1] + ψy[1,ny-1,1:nz-1])/2.0        
    ψz[0,ny-1,1:nz-1] = (ψz[0,ny-2,1:nz-1] + ψz[1,ny-1,1:nz-1])/2.0

    # x direction edges
    ψx[1:nx-1,0,0] = (ψx[1:nx-1,1,0] + ψx[1:nx-1,0,1])/2.0                            # Front-bottom edge
    ψy[1:nx-1,0,0] = (ψy[1:nx-1,1,0] + ψy[1:nx-1,0,1])/2.0        
    ψz[1:nx-1,0,0] = (ψz[1:nx-1,1,0] + ψz[1:nx-1,0,1])/2.0
    ψx[1:nx-1,0,nz-1] = (ψx[1:nx-1,1,nz-1] + ψx[1:nx-1,0,nz-2])/2.0                   # Back-bottom edge
    ψy[1:nx-1,0,nz-1] = (ψy[1:nx-1,1,nz-1] + ψy[1:nx-1,0,nz-2])/2.0         
    ψz[1:nx-1,0,nz-1] = (ψz[1:nx-1,1,nz-1] + ψz[1:nx-1,0,nz-2])/2.0 
    ψx[1:nx-1,ny-1,0] = (ψx[1:nx-1,ny-1,1] + ψx[1:nx-1,ny-2,0])/2.0                   # Front-top edge
    ψy[1:nx-1,ny-1,0] = (ψy[1:nx-1,ny-1,1] + ψy[1:nx-1,ny-2,0])/2.0         
    ψz[1:nx-1,ny-1,0] = (ψz[1:nx-1,ny-1,1] + ψz[1:nx-1,ny-2,0])/2.0 
    ψx[1:nx-1,ny-1,nz-1] = (ψx[1:nx-1,ny-2,nz-1] + ψx[1:nx-1,ny-1,nz-2])/2.0          # Back-top edge
    ψy[1:nx-1,ny-1,nz-1] = (ψy[1:nx-1,ny-2,nz-1] + ψy[1:nx-1,ny-1,nz-2])/2.0        
    ψz[1:nx-1,ny-1,nz-1] = (ψz[1:nx-1,ny-2,nz-1] + ψz[1:nx-1,ny-1,nz-2])/2.0

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

    # print()
    # print(f"ψx at {t:.3f}s post-BCs :")
    # print()
    # print(ψx[:,:,3])

    # print()
    # print(f"ψy at {t:.3f}s post-BCs :")
    # print()
    # print(ψy[:,:,3])

    # print()
    # print(f"Solve for ψz at {t:.3f}s then enforce BCs:")
    # print()
    # print(ψz[:,:,3])


    #---------------------------------------
    # SOLVE FOR THE VELOCITY VECTOR FIELD
    #---------------------------------------
    # using the curl of the vector potential
    # Solve on the internal domain for the velocity field
    u[1:nx-1,1:ny-1,1:nz-1] = (ψz[1:nx-1,2:ny,1:nz-1] - ψz[1:nx-1,0:ny-2,1:nz-1])/(2*dy) + (ψy[1:nx-1,1:ny-1,0:nz-2] - ψy[1:nx-1,1:ny-1,2:nz])/(2*dz)  # First-order central
    v[1:nx-1,1:ny-1,1:nz-1] = (ψx[1:nx-1,1:ny-1,2:nz] - ψx[1:nx-1,1:ny-1,0:nz-2])/(2*dz) + (ψz[0:nx-2,1:ny-1,1:nz-1] - ψz[2:nx,1:ny-1,1:nz-1])/(2*dx)
    w[1:nx-1,1:ny-1,1:nz-1] = (ψy[2:nx,1:ny-1,1:nz-1] - ψy[0:nx-2,1:ny-1,1:nz-1])/(2*dx) + (ψx[1:nx-1,0:ny-2,1:nz-1] - ψx[1:nx-1,2:ny,1:nz-1])/(2*dy)



    # print()
    # print(f"Velocity at {t:.3f}s pre-BCs :")
    # print()
    # print(u[:,:,3])

    # RE-APPLY VELOCITY BOUNDARY CONDITIONS
    u[0,1:ny-1,1:nz-1] = 0.0          # Left wall
    v[0,1:ny-1,1:nz-1] = 0.0
    w[0,1:ny-1,1:nz-1] = 0.0
    u[nx-1,1:ny-1,1:nz-1] = 0.0       # Right wall
    v[nx-1,1:ny-1,1:nz-1] = 0.0
    w[nx-1,1:ny-1,1:nz-1] = 0.0


    u[1:nx-1,1:ny-1,0] = 0.0          # Front wall
    v[1:nx-1,1:ny-1,0] = 0.0
    w[1:nx-1,1:ny-1,0] = 0.0
    u[1:nx-1,1:ny-1,nz-1] = 0.0       # Back wall
    v[1:nx-1,1:ny-1,nz-1] = 0.0
    w[1:nx-1,1:ny-1,nz-1] = 0.0


    u[1:nx-1,0,1:nz-1] = 0.0          # Bottom wall
    v[1:nx-1,0,1:nz-1] = 0.0
    w[1:nx-1,0,1:nz-1] = 0.0
    u[1:nx-1,ny-1,1:nz-1] = Ut        # Top wall
    v[1:nx-1,ny-1,1:nz-1] = 0.0
    w[1:nx-1,ny-1,1:nz-1] = 0.0

    # Velocity edge points


    u[0,1:ny-1,0] = (u[1,1:ny-1,0] + u[0,1:ny-1,1])/2.0                            # Front-left edge
    v[0,1:ny-1,0] = (v[1,1:ny-1,0] + v[0,1:ny-1,1])/2.0
    w[0,1:ny-1,0] = (w[1,1:ny-1,0] + w[0,1:ny-1,1])/2.0
    u[nx-1,1:ny-1,0] = (u[nx-2,1:ny-1,0] + u[nx-1,1:ny-1,1])/2.0                   # Front-right edge
    v[nx-1,1:ny-1,0] = (v[nx-2,1:ny-1,0] + v[nx-1,1:ny-1,1])/2.0
    w[nx-1,1:ny-1,0] = (w[nx-2,1:ny-1,0] + w[nx-1,1:ny-1,1])/2.0
    u[nx-1,1:ny-1,nz-1] = (u[nx-2,1:ny-1,nz-1] + u[nx-1,1:ny-1,nz-2])/2.0          # Back-right edge
    v[nx-1,1:ny-1,nz-1] = (v[nx-2,1:ny-1,nz-1] + v[nx-1,1:ny-1,nz-2])/2.0
    w[nx-1,1:ny-1,nz-1] = (w[nx-2,1:ny-1,nz-1] + w[nx-1,1:ny-1,nz-2])/2.0
    u[0,1:ny-1,nz-1] = (u[1,1:ny-1,nz-1] + u[0,1:ny-1,nz-2])/2.0                   # Back-left edge
    v[0,1:ny-1,nz-1] = (v[1,1:ny-1,nz-1] + v[0,1:ny-1,nz-2])/2.0
    w[0,1:ny-1,nz-1] = (w[1,1:ny-1,nz-1] + w[0,1:ny-1,nz-2])/2.0


    u[0,0,1:nz-1] = (u[1,0,1:nz-1] + u[0,1,1:nz-1])/2.0                            # Bottom-left edge
    v[0,0,1:nz-1] = (v[1,0,1:nz-1] + v[0,1,1:nz-1])/2.0        
    w[0,0,1:nz-1] = (w[1,0,1:nz-1] + w[0,1,1:nz-1])/2.0
    u[nx-1,0,1:nz-1] = (u[nx-2,0,1:nz-1] + u[nx-1,1,1:nz-1])/2.0                      # Bottom-right edge
    v[nx-1,0,1:nz-1] = (v[nx-2,0,1:nz-1] + v[nx-1,1,1:nz-1])/2.0        
    w[nx-1,0,1:nz-1] = (w[nx-2,0,1:nz-1] + w[nx-1,1,1:nz-1])/2.0
    u[nx-1,ny-1,1:nz-1] = (u[nx-2,ny-1,1:nz-1] + u[nx-1,ny-2,1:nz-1])/2.0          # Top-right edge
    v[nx-1,ny-1,1:nz-1] = (v[nx-2,ny-1,1:nz-1] + v[nx-1,ny-2,1:nz-1])/2.0        
    w[nx-1,ny-1,1:nz-1] = (w[nx-2,ny-1,1:nz-1] + w[nx-1,ny-2,1:nz-1])/2.0
    u[0,ny-1,1:nz-1] = (u[0,ny-2,1:nz-1] + u[1,ny-1,1:nz-1])/2.0                   # Top-left edge
    v[0,ny-1,1:nz-1] = (v[0,ny-2,1:nz-1] + v[1,ny-1,1:nz-1])/2.0        
    w[0,ny-1,1:nz-1] = (w[0,ny-2,1:nz-1] + w[1,ny-1,1:nz-1])/2.0


    u[1:nx-1,0,0] = (u[1:nx-1,1,0] + u[1:nx-1,0,1])/2.0                            # Front-bottom edge
    v[1:nx-1,0,0] = (v[1:nx-1,1,0] + v[1:nx-1,0,1])/2.0        
    w[1:nx-1,0,0] = (w[1:nx-1,1,0] + w[1:nx-1,0,1])/2.0
    u[1:nx-1,0,nz-1] = (u[1:nx-1,1,nz-1] + u[1:nx-1,0,nz-2])/2.0                   # Back-bottom edge
    v[1:nx-1,0,nz-1] = (v[1:nx-1,1,nz-1] + v[1:nx-1,0,nz-2])/2.0         
    w[1:nx-1,0,nz-1] = (w[1:nx-1,1,nz-1] + w[1:nx-1,0,nz-2])/2.0 
    u[1:nx-1,ny-1,0] = (u[1:nx-1,ny-1,1] + u[1:nx-1,ny-2,0])/2.0                   # Front-top edge
    v[1:nx-1,ny-1,0] = (v[1:nx-1,ny-1,1] + v[1:nx-1,ny-2,0])/2.0         
    w[1:nx-1,ny-1,0] = (w[1:nx-1,ny-1,1] + w[1:nx-1,ny-2,0])/2.0 
    u[1:nx-1,ny-1,nz-1] = (u[1:nx-1,ny-2,nz-1] + u[1:nx-1,ny-1,nz-2])/2.0          # Back-top edge
    v[1:nx-1,ny-1,nz-1] = (v[1:nx-1,ny-2,nz-1] + v[1:nx-1,ny-1,nz-2])/2.0        
    w[1:nx-1,ny-1,nz-1] = (w[1:nx-1,ny-2,nz-1] + w[1:nx-1,ny-1,nz-2])/2.0

    # Velocity corner points
    u[0,0,0] = (u[1,0,0] + u[0,1,0] + u[0,0,1]) / 3.0                                           # Lower bottom left 
    v[0,0,0] = (v[1,0,0] + v[0,1,0] + v[0,0,1]) / 3.0
    w[0,0,0] = (w[1,0,0] + w[0,1,0] + w[0,0,1]) / 3.0

    u[0,0,nz-1] = (u[0,0,nz-2] + u[1,0,nz-1] + u[0,1,nz-1]) / 3.0                               # Lower back left
    v[0,0,nz-1] = (v[0,0,nz-2] + v[1,0,nz-1] + v[0,1,nz-1]) / 3.0
    w[0,0,nz-1] = (w[0,0,nz-2] + w[1,0,nz-1] + w[0,1,nz-1]) / 3.0

    u[nx-1,0,0] = (u[nx-2,0,0] + u[nx-1,1,0] + u[nx-1,0,1]) / 3.0                               # Lower front right 
    v[nx-1,0,0] = (v[nx-2,0,0] + v[nx-1,1,0] + v[nx-1,0,1]) / 3.0
    w[nx-1,0,0] = (w[nx-2,0,0] + w[nx-1,1,0] + w[nx-1,0,1]) / 3.0

    u[nx-1,0,nz-1] = (u[nx-2,0,nz-1] + u[nx-1,0,nz-2] + u[nx-1,1,nz-1]) / 3.0                   # Lower back right
    v[nx-1,0,nz-1] = (v[nx-2,0,nz-1] + v[nx-1,0,nz-2] + v[nx-1,1,nz-1]) / 3.0
    w[nx-1,0,nz-1] = (w[nx-2,0,nz-1] + w[nx-1,0,nz-2] + w[nx-1,1,nz-1]) / 3.0

    u[0,ny-1,0] = (u[1,ny-1,0] + u[0,ny-2,0] + u[0,ny-1,1]) / 3.0                               # Front top left
    v[0,ny-1,0] = (v[1,ny-1,0] + v[0,ny-2,0] + v[0,ny-1,1]) / 3.0
    w[0,ny-1,0] = (w[1,ny-1,0] + w[0,ny-2,0] + w[0,ny-1,1]) / 3.0

    u[0,ny-1,nz-1] = (u[0,ny-1,nz-2] + u[1,ny-1,nz-1] + u[0,ny-2,nz-1]) / 3.0                   # Back top left
    v[0,ny-1,nz-1] = (v[0,ny-1,nz-2] + v[1,ny-1,nz-1] + v[0,ny-2,nz-1]) / 3.0
    w[0,ny-1,nz-1] = (w[0,ny-1,nz-2] + w[1,ny-1,nz-1] + w[0,ny-2,nz-1]) / 3.0

    u[nx-1,ny-1,0] = (u[nx-2,ny-1,0] + u[nx-1,ny-2,0] + u[nx-1,ny-1,1]) / 3.0                   # Front top right
    v[nx-1,ny-1,0] = (v[nx-2,ny-1,0] + v[nx-1,ny-2,0] + v[nx-1,ny-1,1]) / 3.0
    w[nx-1,ny-1,0] = (w[nx-2,ny-1,0] + w[nx-1,ny-2,0] + w[nx-1,ny-1,1]) / 3.0

    u[nx-1,ny-1,nz-1] = (u[nx-2,ny-1,nz-1] + u[nx-1,ny-1,nz-2] + u[nx-1,ny-2,nz-1]) / 3.0       # Back top right
    v[nx-1,ny-1,nz-1] = (v[nx-2,ny-1,nz-1] + v[nx-1,ny-1,nz-2] + v[nx-1,ny-2,nz-1]) / 3.0
    w[nx-1,ny-1,nz-1] = (w[nx-2,ny-1,nz-1] + w[nx-1,ny-1,nz-2] + w[nx-1,ny-2,nz-1]) / 3.0

    # print()
    # print(f"Solve for u at {t:.3f}s then enforce BCs:")
    # print()
    # print(u[:,:,3])

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
    Cx = u[1:nx-1,1:ny-1,1:nz-1] * (Ωxn[2:nx,1:ny-1,1:nz-1] - Ωxn[0:nx-2,1:nx-1,1:ny-1])/(2*dx)
    Cy = v[1:nx-1,1:ny-1,1:nz-1] * (Ωxn[1:nx-1,2:ny,1:nz-1] - Ωxn[1:nx-1,0:ny-2,1:nz-1])/(2*dy)
    Cz = w[1:nx-1,1:ny-1,1:nz-1] * (Ωxn[1:nx-1,1:ny-1,2:nz] - Ωxn[1:nx-1,1:ny-1,0:nz-2])/(2*dz)

    Dx = (Ωxn[2:nx,1:ny-1,1:nz-1] + Ωxn[0:nx-2,1:ny-1,1:nz-1] - 2*Ωxn[1:nx-1,1:ny-1,1:nz-1])/(dx**2)
    Dy = (Ωxn[1:nx-1,2:ny,1:nz-1] + Ωxn[1:nx-1,0:ny-2,1:nz-1] - 2*Ωxn[1:nx-1,1:ny-1,1:nz-1])/(dy**2)
    Dz = (Ωxn[1:nx-1,1:ny-1,2:nz] + Ωxn[1:nx-1,1:ny-1,0:nz-2] - 2*Ωxn[1:nx-1,1:ny-1,1:nz-1])/(dz**2)

    Ux = Ωxn[1:nx-1,1:ny-1,1:nz-1] * (u[2:nx,1:ny-1,1:nz-1] - u[0:nx-2,1:ny-1,1:nz-1])/(2*dx)
    Uy = Ωyn[1:nx-1,1:ny-1,1:nz-1] * (u[1:nx-1,2:ny,1:nz-1] - u[1:nx-1,0:ny-2,1:nz-1])/(2*dy)
    Uz = Ωzn[1:nx-1,1:ny-1,1:nz-1] * (u[1:nx-1,1:ny-1,2:nz] - u[1:nx-1,1:ny-1,0:nz-2])/(2*dz)

    # The equation
    Ωx[1:nx-1,1:ny-1,1:nz-1] = Ωxn[1:nx-1,1:ny-1,1:nz-1] + dt * (nu * (Dx + Dy + Dz) + Ux + Uy + Uz - (Cx + Cy + Cz))

    #-------------------------------
    # Ω_Y
    #-------------------------------
    Ωy = Ωyn.copy()
    Cx = u[1:nx-1,1:ny-1,1:nz-1] * (Ωyn[2:nx,1:ny-1,1:nz-1] - Ωyn[0:nx-2,1:ny-1,1:nz-1])/(2*dx)
    Cy = v[1:nx-1,1:ny-1,1:nz-1] * (Ωyn[1:nx-1,2:ny,1:nz-1] - Ωyn[1:nx-1,0:ny-2,1:nz-1])/(2*dy)
    Cz = w[1:nx-1,1:ny-1,1:nz-1] * (Ωyn[1:nx-1,1:ny-1,2:nz] - Ωyn[1:nx-1,1:ny-1,0:nz-2])/(2*dz)

    Dx = (Ωyn[2:nx,1:ny-1,1:nz-1] + Ωyn[0:nx-2,1:ny-1,1:nz-1] - 2*Ωyn[1:nx-1,1:ny-1,1:nz-1])/(dx**2)
    Dy = (Ωyn[1:nx-1,2:ny,1:nz-1] + Ωyn[1:nx-1,0:ny-2,1:nz-1] - 2*Ωyn[1:nx-1,1:ny-1,1:nz-1])/(dy**2)
    Dz = (Ωyn[1:nx-1,1:ny-1,2:nz] + Ωyn[1:nx-1,1:ny-1,0:nz-2] - 2*Ωyn[1:nx-1,1:ny-1,1:nz-1])/(dz**2)

    Ux = Ωxn[1:nx-1,1:ny-1,1:nz-1] * (v[2:nx,1:ny-1,1:nz-1] - v[0:nx-2,1:ny-1,1:nz-1])/(2*dx)
    Uy = Ωyn[1:nx-1,1:ny-1,1:nz-1] * (v[1:nx-1,2:ny,1:nz-1] - v[1:nx-1,0:ny-2,1:nz-1])/(2*dy)
    Uz = Ωzn[1:nx-1,1:ny-1,1:nz-1] * (v[1:nx-1,1:ny-1,2:nz] - v[1:nx-1,1:ny-1,0:nz-2])/(2*dz)

    # The equation
    Ωy[1:nx-1,1:ny-1,1:nz-1] = Ωyn[1:nx-1,1:ny-1,1:nz-1] + dt * (nu * (Dx + Dy + Dz) + Ux + Uy + Uz - (Cx + Cy + Cz))

    
    #-------------------------------
    # Ω_Z
    #-------------------------------
    Ωz = Ωzn.copy()
    Cx = u[1:nx-1,1:ny-1,1:nz-1] * (Ωzn[2:nx,1:ny-1,1:nz-1] - Ωzn[0:nx-2,1:ny-1,1:nz-1])/(2*dx)
    Cy = v[1:nx-1,1:ny-1,1:nz-1] * (Ωzn[1:nx-1,2:ny,1:nz-1] - Ωzn[1:nx-1,0:ny-2,1:nz-1])/(2*dy)
    Cz = w[1:nx-1,1:ny-1,1:nz-1] * (Ωzn[1:nx-1,1:ny-1,2:nz] - Ωzn[1:nx-1,1:ny-1,0:nz-2])/(2*dz)

    Dx = (Ωzn[2:nx,1:ny-1,1:nz-1] + Ωzn[0:nx-2,1:ny-1,1:nz-1] - 2*Ωzn[1:nx-1,1:ny-1,1:nz-1])/(dx**2)
    Dy = (Ωzn[1:nx-1,2:ny,1:nz-1] + Ωzn[1:nx-1,0:ny-2,1:nz-1] - 2*Ωzn[1:nx-1,1:ny-1,1:nz-1])/(dy**2)
    Dz = (Ωzn[1:nx-1,1:ny-1,2:nz] + Ωzn[1:nx-1,1:ny-1,0:nz-2] - 2*Ωzn[1:nx-1,1:ny-1,1:nz-1])/(dz**2)

    Ux = Ωxn[1:nx-1,1:ny-1,1:nz-1] * (w[2:nx,1:ny-1,1:nz-1] - w[0:nx-2,1:ny-1,1:nz-1])/(2*dx)
    Uy = Ωyn[1:nx-1,1:ny-1,1:nz-1] * (w[1:nx-1,2:ny,1:nz-1] - w[1:nx-1,0:ny-2,1:nz-1])/(2*dy)
    Uz = Ωzn[1:nx-1,1:ny-1,1:nz-1] * (w[1:nx-1,1:ny-1,2:nz] - w[1:nx-1,1:ny-1,0:nz-2])/(2*dz)

    # The equation
    Ωz[1:nx-1,1:ny-1,1:nz-1] = Ωzn[1:nx-1,1:ny-1,1:nz-1] + dt * (nu * (Dx + Dy + Dz) + Ux + Uy + Uz - (Cx + Cy + Cz))

    # print()
    # print(f"Vorticity at {t:.3f}s pre-BCs :")
    # print()
    # print(Ωz[:,:,3])

    # # Re-apply the vorticity boundary conditions
    Ωx[0,1:ny-1,1:nz-1] = 0.0                                        # Left wall
    Ωy[0,1:ny-1,1:nz-1] = -w[1,1:ny-1,1:nz-1]/dx
    Ωz[0,1:ny-1,1:nz-1] = v[1,1:ny-1,1:nz-1]/dx
    Ωx[nx-1,1:ny-1,1:nz-1] = 0.0                                     # Right wall
    Ωy[nx-1,1:ny-1,1:nz-1] = w[nx-2,1:ny-1,1:nz-1]/dx
    Ωz[nx-1,1:ny-1,1:nz-1] =-v[nx-2,1:ny-1,1:nz-1]/dx


    Ωx[1:nx-1,1:ny-1,0] = -v[1:nx-1,1:ny-1,1]/dz                              # Front wall
    Ωy[1:nx-1,1:ny-1,0] = u[1:nx-1,1:ny-1,1]/dz
    Ωz[1:nx-1,1:ny-1,0] = 0.0
    Ωx[1:nx-1,1:ny-1,nz-1] = v[1:nx-1,1:ny-1,nz-2]/dz                         # Back wall
    Ωy[1:nx-1,1:ny-1,nz-1] = -u[1:nx-1,1:ny-1,nz-2]/dz
    Ωz[1:nx-1,1:ny-1,nz-1] = 0.0
            

    Ωx[1:nx-1,0,1:nz-1] = w[1:nx-1,1,1:nz-1]/dy                               # Bottom wall
    Ωy[1:nx-1,0,1:nz-1] = 0.0
    Ωz[1:nx-1,0,1:nz-1] = -u[1:nx-1,1,1:nz-1]/dy
    Ωx[1:nx-1,ny-1,1:nz-1] = -w[1:nx-1,ny-2,1:nz-1]/dy                        # Top wall
    Ωy[1:nx-1,ny-1,1:nz-1] = 0.0                                     
    Ωz[1:nx-1,ny-1,1:nz-1] = -(Ut - u[1:nx-1,ny-2,1:nz-1])/dy

    # Vorticity edge points

    Ωx[0,1:ny-1,0] = (Ωx[1,1:ny-1,0] + Ωx[0,1:ny-1,1])/2.0                            # Front-left edge
    Ωy[0,1:ny-1,0] = (Ωy[1,1:ny-1,0] + Ωy[0,1:ny-1,1])/2.0
    Ωz[0,1:ny-1,0] = (Ωz[1,1:ny-1,0] + Ωz[0,1:ny-1,1])/2.0
    Ωx[nx-1,1:ny-1,0] = (Ωx[nx-2,1:ny-1,0] + Ωx[nx-1,1:ny-1,1])/2.0                   # Front-right edge
    Ωy[nx-1,1:ny-1,0] = (Ωy[nx-2,1:ny-1,0] + Ωy[nx-1,1:ny-1,1])/2.0
    Ωz[nx-1,1:ny-1,0] = (Ωz[nx-2,1:ny-1,0] + Ωz[nx-1,1:ny-1,1])/2.0
    Ωx[nx-1,1:ny-1,nz-1] = (Ωx[nx-2,1:ny-1,nz-1] + Ωx[nx-1,1:ny-1,nz-2])/2.0          # Back-right edge
    Ωy[nx-1,1:ny-1,nz-1] = (Ωy[nx-2,1:ny-1,nz-1] + Ωy[nx-1,1:ny-1,nz-2])/2.0
    Ωz[nx-1,1:ny-1,nz-1] = (Ωz[nx-2,1:ny-1,nz-1] + Ωz[nx-1,1:ny-1,nz-2])/2.0
    Ωx[0,1:ny-1,nz-1] = (Ωx[1,1:ny-1,nz-1] + Ωx[0,1:ny-1,nz-2])/2.0                   # Back-left edge
    Ωy[0,1:ny-1,nz-1] = (Ωy[1,1:ny-1,nz-1] + Ωy[0,1:ny-1,nz-2])/2.0
    Ωz[0,1:ny-1,nz-1] = (Ωz[1,1:ny-1,nz-1] + Ωz[0,1:ny-1,nz-2])/2.0


    Ωx[0,0,1:nz-1] = (Ωx[1,0,1:nz-1] + Ωx[0,1,1:nz-1])/2.0                            # Bottom-left edge
    Ωy[0,0,1:nz-1] = (Ωy[1,0,1:nz-1] + Ωy[0,1,1:nz-1])/2.0        
    Ωz[0,0,1:nz-1] = (Ωz[1,0,1:nz-1] + Ωz[0,1,1:nz-1])/2.0
    Ωx[nx-1,0,1:nz-1] = (Ωx[nx-2,0,1:nz-1] + Ωx[nx-1,1,1:nz-1])/2.0                   # Bottom-right edge
    Ωy[nx-1,0,1:nz-1] = (Ωy[nx-2,0,1:nz-1] + Ωy[nx-1,1,1:nz-1])/2.0        
    Ωz[nx-1,0,1:nz-1] = (Ωz[nx-2,0,1:nz-1] + Ωz[nx-1,1,1:nz-1])/2.0
    Ωx[nx-1,ny-1,1:nz-1] = (Ωx[nx-2,ny-1,1:nz-1] + Ωx[nx-1,ny-2,1:nz-1])/2.0          # Top-right edge
    Ωy[nx-1,ny-1,1:nz-1] = (Ωy[nx-2,ny-1,1:nz-1] + Ωy[nx-1,ny-2,1:nz-1])/2.0        
    Ωz[nx-1,ny-1,1:nz-1] = (Ωz[nx-2,ny-1,1:nz-1] + Ωz[nx-1,ny-2,1:nz-1])/2.0
    Ωx[0,ny-1,1:nz-1] = (Ωx[0,ny-2,1:nz-1] + Ωx[1,ny-1,1:nz-1])/2.0                   # Top-left edge
    Ωy[0,ny-1,1:nz-1] = (Ωy[0,ny-2,1:nz-1] + Ωy[1,ny-1,1:nz-1])/2.0        
    Ωz[0,ny-1,1:nz-1] = (Ωz[0,ny-2,1:nz-1] + Ωz[1,ny-1,1:nz-1])/2.0


    Ωx[1:nx-1,0,0] = (Ωx[1:nx-1,1,0] + Ωx[1:nx-1,0,1])/2.0                            # Front-bottom edge
    Ωy[1:nx-1,0,0] = (Ωy[1:nx-1,1,0] + Ωy[1:nx-1,0,1])/2.0        
    Ωz[1:nx-1,0,0] = (Ωz[1:nx-1,1,0] + Ωz[1:nx-1,0,1])/2.0
    Ωx[1:nx-1,0,nz-1] = (Ωx[1:nx-1,1,nz-1] + Ωx[1:nx-1,0,nz-2])/2.0                   # Back-bottom edge
    Ωy[1:nx-1,0,nz-1] = (Ωy[1:nx-1,1,nz-1] + Ωy[1:nx-1,0,nz-2])/2.0         
    Ωz[1:nx-1,0,nz-1] = (Ωz[1:nx-1,1,nz-1] + Ωz[1:nx-1,0,nz-2])/2.0 
    Ωx[1:nx-1,ny-1,0] = (Ωx[1:nx-1,ny-1,1] + Ωx[1:nx-1,ny-2,0])/2.0                   # Front-top edge
    Ωy[1:nx-1,ny-1,0] = (Ωy[1:nx-1,ny-1,1] + Ωy[1:nx-1,ny-2,0])/2.0         
    Ωz[1:nx-1,ny-1,0] = (Ωz[1:nx-1,ny-1,1] + Ωz[1:nx-1,ny-2,0])/2.0 
    Ωx[1:nx-1,ny-1,nz-1] = (Ωx[1:nx-1,ny-2,nz-1] + Ωx[1:nx-1,ny-1,nz-2])/2.0          # Back-top edge
    Ωy[1:nx-1,ny-1,nz-1] = (Ωy[1:nx-1,ny-2,nz-1] + Ωy[1:nx-1,ny-1,nz-2])/2.0        
    Ωz[1:nx-1,ny-1,nz-1] = (Ωz[1:nx-1,ny-2,nz-1] + Ωz[1:nx-1,ny-1,nz-2])/2.0

    # Vorticity corner points
    Ωx[0,0,0] = (Ωx[1,0,0] + Ωx[0,1,0] + Ωx[0,0,1]) / 3.0                                       # Front bottom left 
    Ωy[0,0,0] = (Ωy[1,0,0] + Ωy[0,1,0] + Ωy[0,0,1]) / 3.0
    Ωz[0,0,0] = (Ωz[1,0,0] + Ωz[0,1,0] + Ωz[0,0,1]) / 3.0

    Ωx[0,0,nz-1] = (Ωx[0,0,nz-2] + Ωx[1,0,nz-1] + Ωx[0,1,nz-1]) / 3.0                           # Back bottom left
    Ωy[0,0,nz-1] = (Ωy[0,0,nz-2] + Ωy[1,0,nz-1] + Ωy[0,1,nz-1]) / 3.0
    Ωz[0,0,nz-1] = (Ωz[0,0,nz-2] + Ωz[1,0,nz-1] + Ωz[0,1,nz-1]) / 3.0

    Ωx[nx-1,0,0] = (Ωx[nx-2,0,0] + Ωx[nx-1,1,0] + Ωx[nx-1,0,1]) / 3.0                           # Front bottom right
    Ωy[nx-1,0,0] = (Ωy[nx-2,0,0] + Ωy[nx-1,1,0] + Ωy[nx-1,0,1]) / 3.0
    Ωz[nx-1,0,0] = (Ωz[nx-2,0,0] + Ωz[nx-1,1,0] + Ωz[nx-1,0,1]) / 3.0

    Ωx[nx-1,0,nz-1] = (Ωx[nx-2,0,nz-1] + Ωx[nx-1,0,nz-2] + Ωx[nx-1,1,nz-1]) / 3.0               # Back bottom right
    Ωy[nx-1,0,nz-1] = (Ωy[nx-2,0,nz-1] + Ωy[nx-1,0,nz-2] + Ωy[nx-1,1,nz-1]) / 3.0
    Ωz[nx-1,0,nz-1] = (Ωz[nx-2,0,nz-1] + Ωz[nx-1,0,nz-2] + Ωz[nx-1,1,nz-1]) / 3.0

    Ωx[0,ny-1,0] = (Ωx[1,ny-1,0] + Ωx[0,ny-2,0] + Ωx[0,ny-1,1]) / 3.0                           # Front top left
    Ωy[0,ny-1,0] = (Ωy[1,ny-1,0] + Ωy[0,ny-2,0] + Ωy[0,ny-1,1]) / 3.0
    Ωz[0,ny-1,0] = (Ωz[1,ny-1,0] + Ωz[0,ny-2,0] + Ωz[0,ny-1,1]) / 3.0

    Ωx[0,ny-1,nz-1] = (Ωx[0,ny-1,nz-2] + Ωx[1,ny-1,nz-1] + Ωx[0,ny-2,nz-1]) / 3.0               # Back top left
    Ωy[0,ny-1,nz-1] = (Ωy[0,ny-1,nz-2] + Ωy[1,ny-1,nz-1] + Ωy[0,ny-2,nz-1]) / 3.0
    Ωz[0,ny-1,nz-1] = (Ωz[0,ny-1,nz-2] + Ωz[1,ny-1,nz-1] + Ωz[0,ny-2,nz-1]) / 3.0

    Ωx[nx-1,ny-1,0] = (Ωx[nx-2,ny-1,0] + Ωx[nx-1,ny-2,0] + Ωx[nx-1,ny-1,1]) / 3.0               # Front top right
    Ωy[nx-1,ny-1,0] = (Ωy[nx-2,ny-1,0] + Ωy[nx-1,ny-2,0] + Ωy[nx-1,ny-1,1]) / 3.0
    Ωz[nx-1,ny-1,0] = (Ωz[nx-2,ny-1,0] + Ωz[nx-1,ny-2,0] + Ωz[nx-1,ny-1,1]) / 3.0

    Ωx[nx-1,ny-1,nz-1] = (Ωx[nx-2,ny-1,nz-1] + Ωx[nx-1,ny-1,nz-2] + Ωx[nx-1,ny-2,nz-1]) / 3.0   # Back top right
    Ωy[nx-1,ny-1,nz-1] = (Ωy[nx-2,ny-1,nz-1] + Ωy[nx-1,ny-1,nz-2] + Ωy[nx-1,ny-2,nz-1]) / 3.0
    Ωz[nx-1,ny-1,nz-1] = (Ωz[nx-2,ny-1,nz-1] + Ωz[nx-1,ny-1,nz-2] + Ωz[nx-1,ny-2,nz-1]) / 3.0

    # print()
    # print(f"Solve for Ωz at {t:.3f}s then enforce BCs :")
    # print()
    # print(Ωz[:,:,3])

    # Store the solution
    Ωx_sol.append(Ωx.copy())
    Ωy_sol.append(Ωy.copy())
    Ωz_sol.append(Ωz.copy())

    # print()
    # print(f"Ωz_sol at {t:.3f}s post-BCs :")
    # print()
    # print(Ωz[:,:,3])

    # Check for dodgy values
    vort_mag = np.sqrt(Ωx**2 + Ωy**2 + Ωz**2)

    if np.any(np.isinf(vort_mag)) or np.any(np.isnan(vort_mag)):
        print(f"Inf/NaN in vort_mag at t={t}")
        break

    t = t + dt
    its = its + 1

    # Terminal
    if its % 500 == 0:
        print(f'\rits = {its}, t = {t:.3f}, Elapsed: {elapsed_time:.3f} s', end='')
        print()

    # Convergence criteria
    if its > 10 and its % 500 == 0:
        vort_conv = np.linalg.norm(np.vstack([
            np.ravel(Ωx_sol[-1] - Ωx_sol[-2]),
            np.ravel(Ωy_sol[-1] - Ωy_sol[-2]),
            np.ravel(Ωz_sol[-1] - Ωz_sol[-2])
        ]))
        print(f"Ω convergence: {vort_conv:.3f}")

print()

print('-------------------------')

print()
print(f'SUMMARY')

print()

print('-------------------------')
print(f'Run = \'{config['output']}\'')
print(f"nx = {nx}")
print('Fluid Parameters:')
print(f'Ut = {Ut}')
print(f"Re = {Re}")
print('-------------------------')
print('Time Marching Parameters:')
print(f'dt = {dt}')
print(f'conv_crit = {conv_crit}')
print('-------------------------')
print('Poisson Parameters:')
print(f"β = {β}")
print(f'itmax = {itmax}')
print(f"tol = {tol}")
print('-------------------------')
print(f'Physical end time = {t:.3f} mins')
print(f'Wall clock time = {elapsed_time:.3f} s')
print(f'Iterations = {its}')
print()
print('Done.')

save_dir = '/home/brierleyajb/Documents/incompressible_repo/vector_psi_omega/results'
os.makedirs(save_dir, exist_ok=True)

#------------------
# YX Centreplane
#------------------

mid = nx // 2
aspect_ratio = Lx / Ly
subplot_height = 5
subplot_width = subplot_height * aspect_ratio
total_width = 3 * subplot_width

plt.figure(figsize=(total_width, subplot_height))

# Subplot 1: u-velocity
plt.subplot(1, 3, 1)
plt.contour(u[:, :, mid].T, origin='lower', extent=[0, Lx, 0, Ly], cmap='viridis',levels=100)
plt.title('u-velocity (Z=nx/2)')
plt.colorbar()
plt.xlabel('X')
plt.ylabel('Y')

# Subplot 2: Vorticity Magnitude
plt.subplot(1, 3, 2)
vort_mag = np.sqrt(Ωx**2 + Ωy**2 + Ωz**2)
plt.contour(vort_mag[:, :, mid].T, origin='lower', extent=[0, Lx, 0, Ly], cmap='plasma',levels=100)
plt.title('Vorticity Magnitude (Z=nx/2)')
plt.colorbar()
plt.xlabel('X')
plt.ylabel('Y')

# Subplot 3: Vector Potential Magnitude
plt.subplot(1, 3, 3)
psi_mag = np.sqrt(ψx**2 + ψy**2 + ψz**2)
plt.contour(psi_mag[:, :, mid].T, origin='lower', extent=[0, Lx, 0, Ly], cmap='plasma',levels=100)
plt.title('Vector Potential Magnitude (Z=nx/2)')
plt.colorbar()
plt.xlabel('X')
plt.ylabel('Y')

plt.suptitle(f'tend = {t:.2f}, Re = {Re:.0f}, nx = {nx}, dt = {dt:.2f}, tol = {tol:.1f}, conv={vort_conv:.0e}, t: {(elapsed_time):.2f} s')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, f'{config['output']}_YX.png'), dpi=300, bbox_inches='tight')

#-----------------------------------------------
# EXTRACT CENTREPLANE-CENTRELINE VELOCITIES
#-----------------------------------------------

# EXTRACTION
u_centreline = np.flip(u[nx//2,:,nz//2]/Ut)
y = np.linspace(Ly,0,ny)

# CSV
data = np.column_stack((y, u_centreline))
np.savetxt(os.path.join(save_dir, f'{config['output']}_u.csv'), data, fmt='%.1g', delimiter=',', header='y,u_centreline', comments='')

# PLOT
csv_data = pd.read_csv('lit_data/2016_chen_all_results.csv')
y_csv_100 = csv_data['Re=100'] 
y_csv_400 = csv_data['Re=400'] 
y_csv_1000 = csv_data['Re=1000']
u_csv_100 = csv_data['x100']
u_csv_400 = csv_data['x400']
u_csv_1000 = csv_data['x1000']
plt.figure()
plt.plot(y,u_centreline,'--m', label=f'Re={Re:.0f}')
plt.plot(y_csv_100, u_csv_100, '-b', label='Chen Re=100')
plt.plot(y_csv_400, u_csv_400, '-r', label='Chen Re=400')
plt.plot(y_csv_1000, u_csv_1000, '-k', label='Chen Re=1000')
plt.xlabel('y')
plt.ylabel('Centreline Velocity')
plt.legend()
plt.suptitle(f'{config['output']}')
# Incorporate simulation settings in the file name
plt.title(f'tend={t:.2f}, Re={Re:.0f}, nx={nx}, dt={dt}, tol={tol}, conv={vort_conv:.1f}, t={(elapsed_time):.2f}s')
# Save figure to the results folder
plt.savefig(os.path.join(save_dir, f'{config['output']}_u.png'), dpi=300, bbox_inches='tight')

