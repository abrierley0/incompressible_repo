import numpy as np
import matplotlib.pyplot as plt
from pyevtk.hl import gridToVTK

# Streamfunction-Vorticity Formulation



# Grid parameters
nx = 64
ny = 64
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
Ut = 5.0
beta = 1.95
nu = 0.05
Re = Ut*Lx/nu
print(f"Re = {Re}")


#------------------------------
# SPECIFY STREAMFUNCTION BCS
#-----------------------------

for i in range(1,nx-1):
    ψ0[i,0] = 0.0
    ψ0[i,ny-1] = 0.0
for j in range(1,ny-1):
    ψ0[nx-1,j] = 0.0
    ψ0[0,j] = Ut*dy

# Corner points
ψ0[0,0] = (ψ0[0,1] + ψ0[1,0])/2.0  # Top left
ψ0[0,ny-1] = (ψ0[0,ny-2] + ψ0[1,ny-1])/2.0  # Top right
ψ0[ny-1,0] = (ψ0[ny-2,0] + ψ0[ny-1,1])/2.0  # Bottom left
ψ0[ny-1,nx-1] = (ψ0[ny-1,nx-2] + ψ0[ny-2,nx-1])/2.0  # Bottom right

print("ψ with BCs is:")
np.set_printoptions(linewidth=1000, precision=2, suppress=True)
print(ψ0)



#----------------------------------------
# SPECIFY VORTICITY BOUNDARY CONDITIONS
#----------------------------------------

for i in range(1,nx-1):
    ω0[i,0] = 2.0*(ψ0[i,0] - ψ0[i,1])/(dx**2)
    ω0[i,nx-1] = 2.0*(ψ0[i,nx-1] - ψ0[i,nx-2])/(dx**2)
for j in range(1,ny-1):
    ω0[ny-1,j] = 2.0*(ψ0[ny-1,j] - ψ0[ny-2,j])/(dy**2)
    ω0[0,j] = 2.0*(ψ0[0,j] - ψ0[1,j])/(dy**2) - 2.0*Ut/dy

# Vorticity at corner points
ω0[0,0] = (ω0[0,1] + ω0[1,0])/2.0  # Top left
ω0[ny-1,0] = (ω0[ny-2,0] + ω0[ny-1,1])/2.0  # Bottom left
ω0[ny-1,nx-1] = (ω0[ny-1,nx-2] + ω0[ny-2,nx-1])/2.0  # Bottom right
ω0[0,nx-1] = (ω0[0,nx-2] + ω0[1,ny-1])/2.0  # Top right

print("ω0 with BCs is:")
np.set_printoptions(linewidth=1000, precision=2, suppress=True)
print(ω0)




# Time-marching parameters
t = 0.0
itmax = 100  # Changed from 10 to 100
tol = 1e-3
dt = min(0.25*dx*dx/nu, 4*nu/Ut/Ut)
print(f"dt = {dt}")
tend = 1.0
print(f"tend = {tend}")



# SOLUTION STORAGE
ψ_sol = []
ψ_sol.append(ψ0)
ω_sol = []
ω_sol.append(ω0)



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
                #ψ_k = ψ.copy()
                # One problem was an unbracketed denominator in the multiplier below
                ψ[i,j] = (beta/(2*(dx**2+dy**2))) * ((dx**2*dy**2)*ωn[i,j] + dy**2*(ψ[i+1,j] + ψ[i-1,j]) + dx**2*(ψ[i,j+1]+ψ[i,j-1])) + (1-beta)*ψ[i,j]
                #ψ[i,j] = beta * (dy**2 * (ψ[i+1,j] + ψ[i-1,j]) + dx**2 * (ψ[i,j+1] + ψ[i,j-1]) + dx**2 * dy**2 * ω[i,j]) / (2 * (dx**2 + dy**2)) + (1 - beta) * ψ[i,j]
        it = it + 1
        err = np.linalg.norm(ψ.ravel() - ψ_k.ravel())
        if it % 10 == 0:
            print(f"iteration = {it}")
            print(f"Error is {err}")

    # Update the boundary conditions, though perhaps not necessary, only in sense of ψ over ψ0 (actual values don't change; but note they do for vorticity)
    for i in range(1,nx-1):
        ψ[i,0] = 0.0
        ψ[i,ny-1] = 0.0
    for j in range(1,ny-1):
        ψ[0,j] = Ut*dy
        ψ[nx-1,j] = 0.0

    # Corner points
    ψ[0,0] = (ψ[0,1] + ψ[1,0])/2.0  # Top left
    ψ[0,ny-1] = (ψ[0,ny-2] + ψ[1,ny-1])/2.0  # Top right
    ψ[ny-1,0] = (ψ[ny-2,0] + ψ[ny-1,1])/2.0  # Bottom left
    ψ[ny-1,nx-1] = (ψ[ny-1,nx-2] + ψ[ny-2,nx-1])/2.0  # Bottom right

    ψ_sol.append(ψ.copy())

#     #----------------------------
#     # SOLVE FOR THE VELOCITY FIELD
#     #----------------------------
#     for i in range(1,nx-2):
#         for j in range(1,ny-2):
#             u0[i,j] = (ψ[i,j+1] - ψ[i,j-1])/2.0/dy
#             v0[i,j] = (ψ[i-1,j] - ψ[i+1,j])/2.0/dx


    #---------------------------------------
    # SOLVE THE 2D VORTICITY TRANSPORT EQ.
    #---------------------------------------

    ω = ωn.copy()  # Copy previous time step vorticity
    # Solve on the internal domain in the current time step for vorticity
    for i in range(1,nx-1):
        for j in range(1,ny-1):
            Cx = (ψ[i,j+1] - ψ[i,j-1])/2.0/dy * (ωn[i+1,j] - ωn[i-1,j])/2.0/dx
            Cy = -(ψ[i+1,j] - ψ[i-1,j])/2.0/dx * (ωn[i,j+1] - ωn[i,j-1])/2.0/dy
            Dx = (ωn[i+1,j] + ωn[i-1,j] - 2*ωn[i,j])/dx**2
            Dy = (ωn[i,j+1] + ωn[i,j-1] - 2*ωn[i,j])/dy**2

            ω[i,j] = dt * (nu*(Dx + Dy) - Cx - Cy) + ωn[i,j]

    # Enforce (or update?) vorticity boundary conditions
    for i in range(1,nx-1):
        ω[i,0] = 2.0*(ψ[i,0] - ψ[i,1])/(dx**2)
        ω[i,ny-1] = 2.0*(ψ[i,nx-1] - ψ[i,nx-2])/(dx**2)
    for j in range(1,ny-1):
        ω[ny-1,j] = 2.0*(ψ[ny-1,j] - ψ[ny-2,j])/(dy**2)
        ω[0,j] = 2.0*(ψ[0,j] - ψ[1,j])/(dy**2) - 2.0*Ut/dy

    # Vorticity at corner points
    ω[0,0] = (ω[0,1] + ω[1,0])/2.0  # Top left
    ω[ny-1,0] = (ω[ny-2,0] + ω[ny-1,1])/2.0  # Bottom left
    ω[ny-1,nx-1] = (ω[ny-1,nx-2] + ω[ny-2,nx-1])/2.0  # Bottom right
    ω[0,nx-1] = (ω[0,nx-2] + ω[1,ny-1])/2.0  # Top right


    ω_sol.append(ω.copy())

#     #------------------------------
#     # SOLVE THE PRESSURE-POISSON 
#     #------------------------------

    # Save to VTK every 1000 steps
    if step % 1000 == 0:
        x = np.linspace(0, Lx, nx)
        y = np.linspace(0, Ly, ny)
        z = np.zeros(1)  # 2D simulation, z = 0
        # Reshape arrays to (nx, ny, 1) for VTK
        ψ_vtk = np.expand_dims(ψ, axis=-1)
        ω_vtk = np.expand_dims(ω, axis=-1)
        # Save to VTK
        gridToVTK(
            f"./flow_{step:04d}",
            x, y, z,
            pointData={
                "streamfunction": ψ_vtk,  # Transpose for VTK's row-major order
                "vorticity": ω_vtk,
            }
        )
        print(f"Saved VTK file: flow_{step:04d}.vtk, t = {t:.4f}, max |ψ| = {np.max(np.abs(ψ)):.4f}")

    t = t + dt
    step = step + 1
    print(f"t = {t}")

print(Re)

X, Y = np.meshgrid(np.linspace(0, Lx, nx), np.linspace(0, Ly, ny))
plt.contourf(X, Y, ψ_sol[-1])
plt.colorbar()
plt.xlabel('x')
plt.ylabel('y')
plt.gca().invert_yaxis()
plt.savefig("psi_sol.png")
plt.close()

