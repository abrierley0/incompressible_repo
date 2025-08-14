import numpy as np

nx = 21
ny = 21
Lx = 1.0
Ly = 1.0

dx = Lx/(nx-1)
dy = Ly/(ny-1)

ψ0 = np.zeros([nx,ny])
Ω0 = np.zeros([nx,ny])

ψ[0,1:ny-1] = 