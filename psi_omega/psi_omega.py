import numpy as np
import matplotlib.pyplot as plt
from pyevtk.hl import gridToVTK

# STREAMFUNCTION VORTICITY FORMULATION
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



# Grid parameters
nx = 26  # imax
ny = 26  # jmax
Lx = 1.0
Ly = 1.0
dx = Lx/(nx-1)
dy = Ly/(ny-1)



# Initialise arrays
ψ0 = np.zeros([nx,ny])
u0 = np.zeros([nx,ny])
v0 = np.zeros([nx,ny])
ω0 = np.zeros([nx,ny])



# Simulation parameters
Ut = 3.2
nu = 0.05
Re = Ut*Lx/nu

print()
print(f"REYNOLDS' NUMBER = {Re}")
print()



#                                TOP WALL
#                 (i=ny-1,j=0)              (i=ny-1,j=nx-1)
#                      *-------------------------*
#                      /                         /
#                      /                         /
#                      /                         /
#                      /                         /
#     LEFT WALL        /                         /         RIGHT WALL
#                      /                         /
#                      /                         /
#                      /                         /
#                      /                         /
#                      *-------------------------*
#                 (i=0, j=0)                 (i=0,j=nx-1)
#
#                                BOTTOM WALL
#
# NOTE: Physical layout



#------------------------------
# SPECIFY STREAMFUNCTION BCS
#-----------------------------

for i in range(1,nx-1):
    ψ0[i,0] = 0.0                                                # Left wall
    ψ0[i,ny-1] = 0.0                                             # Right wall
for j in range(1,ny-1):    
    ψ0[nx-1,j] = 0.0                                             # Top wall
    ψ0[0,j] = 0.0                                                # Bottom wall
    
# Corner points    
ψ0[0,0] = (ψ0[0,1] + ψ0[1,0])/2.0                                # Bottom left
ψ0[0,ny-1] = (ψ0[0,ny-2] + ψ0[1,ny-1])/2.0                       # Bottom right
ψ0[ny-1,0] = (ψ0[ny-2,0] + ψ0[ny-1,1])/2.0                       # Top left
ψ0[ny-1,nx-1] = (ψ0[ny-1,nx-2] + ψ0[ny-2,nx-1])/2.0              # Top right

print("ψ0 with BCs is:")
np.set_printoptions(linewidth=1000, precision=2, suppress=True)
print(ψ0)
print()



#----------------------------------------
# SPECIFY VORTICITY BOUNDARY CONDITIONS
#----------------------------------------
# using the Pletcher book

for i in range(1,nx-1):
    ω0[i,0] = 2.0*(ψ0[i,0] - ψ0[i,1])/(dx**2)                    # Left wall
    ω0[i,ny-1] = 2.0*(ψ0[i,ny-1] - ψ0[i,ny-2])/(dx**2)           # Right wall
for j in range(1,ny-1):
    ω0[ny-1,j] = 2.0*(ψ0[ny-1,j] - ψ0[ny-2,j] - Ut*dy)/(dy**2)   # Top wall
    ω0[0,j] = 2.0*(ψ0[0,j] - ψ0[1,j])/(dy**2)                    # Bottom wall

# Vorticity at corner points
ω0[0,0] = (ω0[0,1] + ω0[1,0])/2.0                                # Bottom left
ω0[ny-1,0] = (ω0[ny-2,0] + ω0[ny-1,1])/2.0                       # Top left
ω0[ny-1,nx-1] = (ω0[ny-1,nx-2] + ω0[ny-2,nx-1])/2.0              # Top right
ω0[0,nx-1] = (ω0[0,nx-2] + ω0[1,ny-1])/2.0                       # Bottom right

# Velocity on top wall appears in the vorticity boundary condition (?)

print("ω0 with BCs is:")
np.set_printoptions(linewidth=1000, precision=2, suppress=True)
print(ω0)




# Time-marching parameters
t = 0.0
itmax = 100 
tol = 1e-4
β = 1.7
dt = min(0.25*dx*dx/nu, 4*nu/Ut/Ut)
#dt = min(0.1 * dx / Ut, 0.25 * dx * dx / nu)
print(f"dt = {dt}")
tend = 30.0
print(f"tend = {tend}")



# SOLUTION STORAGE
ψ_sol = []
ψ_sol.append(ψ0)
ω_sol = []
ω_sol.append(ω0)
u_sol = []
u_sol.append(u0)
v_sol = []
v_sol.append(v0)



# Start main time loop
step = 0
while t < tend:
    #-------------------------------------------
    # SOLVE THE STREAMFUNCTION-POISSON EQUATION
    #-------------------------------------------
    it = 0
    err = 1e5
    ωn = ω_sol[-1].copy()
    ψ = ψ_sol[-1].copy()



    while it < itmax and err > tol:
        # Loop over the grid
        ψ_k = ψ.copy()
        # Solve for psi on the internal domain in the present time step
        for i in range(1,nx-1):
            for j in range(1,ny-1):
                ψ[i,j] = (β/(2*(dx**2+dy**2))) * ((dx**2*dy**2)*ωn[i,j] + dy**2*(ψ[i+1,j] + ψ[i-1,j]) + dx**2*(ψ[i,j+1]+ψ[i,j-1])) + (1-β)*ψ[i,j]
        it = it + 1
        err = np.linalg.norm(ψ.ravel() - ψ_k.ravel())
        #if it % 10 == 0:
        print(f"iteration = {it}")
        print(f"Error is {err}")




    # Update streamfunction boundary values
    for i in range(1,nx-1):
        ψ[i,0] = 0.0                                              # Left wall
        ψ[i,ny-1] = 0.0                                           # Right wall
    for j in range(1,ny-1):    
        ψ[nx-1,j] = 0.0                                           # Top wall
        ψ[0,j] = 0.0                                              # Bottom wall
        
    # Streamfunction corner points    
    ψ[0,0] = (ψ[0,1] + ψ[1,0])/2.0                                # Bottom left
    ψ[0,ny-1] = (ψ[0,ny-2] + ψ[1,ny-1])/2.0                       # Bottom right
    ψ[ny-1,0] = (ψ[ny-2,0] + ψ[ny-1,1])/2.0                       # Top left
    ψ[ny-1,nx-1] = (ψ[ny-1,nx-2] + ψ[ny-2,nx-1])/2.0              # Top right

    # Update the streamfunction solution
    ψ_sol.append(ψ.copy())

    #----------------------------------
    # SOLVE FOR THE VELOCITY VECTOR FIELD
    #---------------------------------

    #---------------------------------------
    # SOLVE THE 2D VORTICITY TRANSPORT EQ.
    #---------------------------------------

    ω = ωn.copy()  # Copy previous time step vorticity field
    # Solve on the internal domain in the current time step for the vorticity field
    for i in range(1,nx-1):
        for j in range(1,ny-1):
            Cx = (ψ[i,j+1] - ψ[i,j-1])/2.0/dy * (ωn[i+1,j] - ωn[i-1,j])/2.0/dx
            Cy = -(ψ[i+1,j] - ψ[i-1,j])/2.0/dx * (ωn[i,j+1] - ωn[i,j-1])/2.0/dy
            Dx = (ωn[i+1,j] + ωn[i-1,j] - 2*ωn[i,j])/dx**2
            Dy = (ωn[i,j+1] + ωn[i,j-1] - 2*ωn[i,j])/dy**2

            ω[i,j] = dt * (nu*(Dx + Dy) - Cx - Cy) + ωn[i,j]

    # Update vorticity field boundary values
    for i in range(1,nx-1):
        ω[i,0] = 2.0*(ψ[i,0] - ψ[i,1])/(dx**2)                    # Left wall
        ω[i,ny-1] = 2.0*(ψ[i,ny-1] - ψ[i,ny-2])/(dx**2)           # Right wall
    for j in range(1,ny-1):
        ω[ny-1,j] = 2.0*(ψ[ny-1,j] - ψ[ny-2,j] - Ut*dy)/(dy**2)   # Top wall
        ω[0,j] = 2.0*(ψ[0,j] - ψ[1,j])/(dy**2)                    # Bottom wall

    # Vorticity at corner points
    ω[0,0] = (ω[0,1] + ω[1,0])/2.0                                # Bottom left
    ω[ny-1,0] = (ω[ny-2,0] + ω[ny-1,1])/2.0                       # Top left
    ω[ny-1,nx-1] = (ω[ny-1,nx-2] + ω[ny-2,nx-1])/2.0              # Top right
    ω[0,nx-1] = (ω[0,nx-2] + ω[1,ny-1])/2.0                       # Bottom right

    ω_sol.append(ω.copy())

    #-----------------------------------------------------
    # SOLVE THE PRESSURE-POISSON FOR THE PRESSURE FIELD
    #-----------------------------------------------------


#     # Save in ParaView .vtk format
#     if step % 10 == 0:
#         x = np.linspace(0, Lx, nx)
#         y = np.linspace(0, Ly, ny)
#         z = np.zeros(1)  # 2D simulation, z = 0
#         # Reshape arrays to (nx, ny, 1) for VTK
#         ψ_vtk = np.expand_dims(ψ, axis=-1)
#         ω_vtk = np.expand_dims(ω, axis=-1)
#         u_vtk = np.expand_dims(u, axis=-1)
#         v_vtk = np.expand_dims(v, axis=-1)
#         # Save to VTK
#         gridToVTK(
#             f"./flow_{step:04d}",
#             x, y, z,
#             pointData={
#                 "streamfunction": ψ_vtk,  # Transpose for VTK's row-major order
#                 "vorticity": ω_vtk,
#                 "u": u_vtk,
#                 "v": v_vtk,
#             }
#         )
#         print(f"Saved VTK file: flow_{step:04d}.vtk, t = {t:.4f}, max |ψ| = {np.max(np.abs(ψ)):.4f}")

    t = t + dt
    step = step + 1
    print(f"t = {t}")

# INSPECT FINAL MATRICES
#print(ψ)
#print(ω)

#IMAGES
X, Y = np.meshgrid(np.linspace(0, Lx, nx), np.linspace(0, Ly, ny))
plt.contourf(X, Y, ψ_sol[-1])
plt.colorbar()
plt.xlabel('x')
plt.ylabel('y')
#plt.gca().invert_yaxis()
plt.title(f't = {t}, Re = {Re}, tol = {tol}, nx = {nx}, β = {β}, itmax = {itmax}')
#plt.text(X, Y, 'text')
plt.savefig("psi_sol.png")
plt.close()

X, Y = np.meshgrid(np.linspace(0, Lx, nx), np.linspace(0, Ly, ny))
plt.contourf(X, Y, ω_sol[-1])
plt.colorbar()
plt.xlabel('x')
plt.ylabel('y')
#plt.gca().invert_yaxis()
plt.title(f't = {t}, Re = {Re}, tol = {tol}, nx = {nx}, β = {β}, itmax = {itmax}')
#plt.text(X, Y, 'text')
plt.savefig("omega_sol.png")
plt.close()

