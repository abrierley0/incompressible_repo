

#############################################
# PRESSURE-POISSON METHOD OF HARLOW & WELCH #
#############################################

# Written by A. Brierley
# 16/12/2025
# Bedford, Bedfordshire, UK

import numpy as np

np.set_printoptions(precision=0, suppress=True, linewidth=1000)


#########################
# Simulation parameters #
#########################

# Grid
L = 1.0
H = 1.0
nx = 10
ny = 10
dx = L/(nx-1)
dy = H/(ny-1)

# Fluid


#####################
# Initialise arrays #
#####################

X = np.zeros((nx,ny))
Y = np.zeros((nx,ny))
u = np.zeros((nx,ny))
v = np.zeros((nx,ny))
w = np.zeros((nx,ny))
p = np.zeros((nx,ny))


###############################
# DEFINE A RECTANGULAR DOMAIN #
###############################

x = 0
y = 0
for i in range(1,nx):
    for j in range(1, ny):
        x = x + dx
        y = y + dy
        X[i,j] = x
        Y[i,j] = y

print("X:")
print(X)
print("Y:")
print(Y)




#######################
# BOUNDARY CONDITIONS #
#######################

# Left wall
for i in range(1,nx-1):
    u[i,0] = 0.0
    v[i,0] = 0.0

# Right wall
for i in range(1,nx-1):
    u[nx-1,0] = 0.0
    v[nx-1,0] = 0.0

# Bottom wall 
for j in range(1,ny-1):
    u[i,ny-1] = 0.0
    v[i,ny-1] = 0.0

# Top wall
for j in range(1,ny-1):
    u[0,j] = 1.0
    v[0,j] = 0.0

#################
# CORNER POINTS #
#################

print(f"u[nx,1]: {u[nx-1,0]}")

u[0,0] = (u[1,0] + u[0,1])/2.0
u[nx-1,0] = (u[nx-2,0] + u[nx-1,1])/2.0
u[0,ny-1] = (u[0,ny-2] + u[1,ny-1])/2.0
u[nx-1,ny-1] = (u[nx-1,ny-2] + u[nx-2,ny-1])/2.0

v[0,0] = (v[1,0] + v[0,1])/2.0
v[nx-1,0] = (v[nx-2,0] + v[nx-1,1])/2.0
v[0,ny-1] = (v[0,ny-1] + v[1,ny-1])/2.0
v[nx-1,ny-1] = (v[nx-1,ny-1] + v[nx-1,ny-1])/2.0

print(f"u[nx-1,1]: {u[nx-1,1]}")

print("u:")
print(u)

print("v:")
print(v)


