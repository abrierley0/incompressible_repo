import numpy as np

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
ψ0 = np.zeros([nx,ny,nz])
ω0 = np.zeros([nx,ny,nz])

print(f"ψ0 is : ")
print()
print(ψ0)



# Time-marching parameters
tend = 5.0
tol = 1e-3
err = 1e5
itmax = 100
β = 1.7
dt = min(0.25*dx*dx/Ϟ, 4*Ϟ/Ut/Ut)


# Create solution storage
ψ_sol = []
ψ_sol.append(ψ0)
ω_sol = []
ω_sol.append(ω0)


# Start main time loop
t = 0
ψ = ψ0
ω = ω0
while t < tend:

    it = 0
    ψ = ψ_sol[-1].copy()
    ω = ω_sol[-1].copy()
    # Solve three Poisson equations for the vector-potential field
    while it < itmax and err > tol:
        for i in range(1,nx-1):
            for j in range(1,ny-1):
                for k in range(1,nz-1):
                    ψ[i,j,k] = (β / (2*(dx**2*dz**2 + dy**2*dz**2 + dx**2*dy**2))) * (dx**2*dy**2*dz**2*ω[i,j,k] + dy**2*dz**2*(ψ[i+1,j,k]+ψ[i-1,j,k]) + dx**2*dz**2*(ψ[i,j+1,k]+ψ[i,j-1,k]) + dx**2*dy**2*(ψ[i,j,k+1] + ψ[i,j,k+1])) + (1 - β) * ψ[i,j,k]

    ψ_sol.append(ψ)

    # Solve the 3D vorticity transport equation with the vortex stretching term
    #
    #
    #

    ω_sol.append(ω)

    t = t + dt
    it = it + 1


