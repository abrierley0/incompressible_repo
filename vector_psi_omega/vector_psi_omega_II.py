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
ψx_sol = []
ψx_sol.append(ψ0)
ψy_sol = []
ψy_sol.append(ψ0)
ψz_sol = []
ψz_sol.append(ψ0)
ωx_sol = []
ωx_sol.append(ω0)
ωy_sol = []
ωy_sol.append(ω0)
ωz_sol = []
ωz_sol.append(ω0)


# Start main time loop
t = 0
while t < tend:

    #---------------------------------------------------------------------------
    # SOLVE THREE VECTOR-POTENTIAL POISSON EQUATIONS USING GAUSS-SEIDEL S.O.R
    #---------------------------------------------------------------------------

    # POISSON SOLVER FOR ψ_x
    it = 0
    ψx = ψx_sol[-1].copy()
    ωx = ωx_sol[-1].copy()
    while it < itmax and err > tol:
        for i in range(1,nx-1):
            for j in range(1,ny-1):
                for k in range(1,nz-1):
                    ψx[i,j,k] = (β / (2*(dx**2*dz**2 + dy**2*dz**2 + dx**2*dy**2))) * (dx**2*dy**2*dz**2*ωx[i,j,k] + dy**2*dz**2*(ψx[i+1,j,k]+ψx[i-1,j,k]) + dx**2*dz**2*(ψx[i,j+1,k]+ψx[i,j-1,k]) + dx**2*dy**2*(ψx[i,j,k+1] + ψx[i,j,k+1])) + (1 - β) * ψx[i,j,k]
        it = it + 1
    ψx_sol.append(ψx)

    # POISSON SOLVER FOR ψ_y
    it = 0
    ψy = ψy_sol[-1].copy()
    ωy = ωy_sol[-1].copy()
    while it < itmax and err > tol:
        for i in range(1,nx-1):
            for j in range(1,ny-1):
                for k in range(1,nz-1):
                    ψy[i,j,k] = (β / (2*(dx**2*dz**2 + dy**2*dz**2 + dx**2*dy**2))) * (dx**2*dy**2*dz**2*ωy[i,j,k] + dy**2*dz**2*(ψy[i+1,j,k]+ψy[i-1,j,k]) + dx**2*dz**2*(ψy[i,j+1,k]+ψy[i,j-1,k]) + dx**2*dy**2*(ψy[i,j,k+1] + ψy[i,j,k+1])) + (1 - β) * ψy[i,j,k]
        it = it + 1
    ψy_sol.append(ψy)

    # POISSON SOLVER FOR ψ_z
    it = 0 
    ψz = ψz_sol[-1].copy()
    ωz = ωz_sol[-1].copy()
    while it < itmax and err > tol:
        for i in range(1,nx-1):
            for j in range(1,ny-1):
                for k in range(1,nz-1):
                    ψz[i,j,k] = (β / (2*(dx**2*dz**2 + dy**2*dz**2 + dx**2*dy**2))) * (dx**2*dy**2*dz**2*ωz[i,j,k] + dy**2*dz**2*(ψz[i+1,j,k]+ψz[i-1,j,k]) + dx**2*dz**2*(ψz[i,j+1,k]+ψz[i,j-1,k]) + dx**2*dy**2*(ψz[i,j,k+1] + ψz[i,j,k+1])) + (1 - β) * ψz[i,j,k]
        it = it + 1
    ψz_sol.append(ψz)

    #---------------------------------------------------------------------------------
    # SOLVE THE 3D VORTICITY TRANSPORT EQUATION INCLUDING THE VORTEX STRETCHING TERM
    #---------------------------------------------------------------------------------

    #ω_sol.append(ω)

    t = t + dt


