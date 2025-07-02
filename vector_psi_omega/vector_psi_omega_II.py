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
ψx0 = np.zeros([nx,ny,nz])
ψy0 = np.zeros([nx,ny,nz])
ψz0 = np.zeros([nx,ny,nz])

Ωx0 = np.zeros([nx,ny,nz])
Ωy0 = np.zeros([nx,ny,nz])
Ωz0 = np.zeros([nx,ny,nz])

print(f"ψx0 is : ")
print()
print(ψx0)

print(f"Ωx0 is : ")
print()
print(Ωx0)




#---------------------------------------------
# VECTOR-POTENTIAL (ψ) BOUNDARY CONDITIONS
#---------------------------------------------



#----------------------------------------------
# VORTICITY VECTOR FIELD BOUNDARY CONDITIONS
#----------------------------------------------





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






    #---------------------------------------------------------------------------------
    # SOLVE THE 3D VORTICITY TRANSPORT EQUATION INCLUDING THE VORTEX STRETCHING TERM
    #---------------------------------------------------------------------------------
    # We solve three equations, one for each vorticity component

    #Ω_sol.append(Ω)





    t = t + dt


