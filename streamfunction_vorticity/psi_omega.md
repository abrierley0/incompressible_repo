Streamfunction-Vorticity Formulation
====================================

#### Pseudo-code for the program
```python
# Define cavity dimensions, cell count, and grid spacing
# Set the lid speed
# Create a square array with storage for psi solution values
# Create a square array with storage for omega solution values
# Set psi on the wall as zero to coincide with zero velocity of the walls
# Apply vorticity boundary conditions 
    # based on the rigid wall conditions of no-slip and no-penetration
    # u=0, v=0
    # This implies psi=0
```

#### Aspects of the Theory

------------------------------------------

###### Vorticity Vector Field
The vorticity vector field is,
$\vec{\Omega} = \nabla \times \vec{u}$.
At every point in the spatial and time domain there is a particular vorticity vector:
$\vec{\Omega} = \vec{\Omega}(x,y,z;t)$.
In three dimensions, there are three scalar components of the vorticity vector which are generally non-zero,
$\Omega_x, \Omega_y, \Omega_z \ne 0$,
where each scalar component is representing ths strength and direction of rotation about the rotation axis at that point in space and time:
$\vec{\Omega}(x,y,z;t) = \Omega_x \vec{e}_x + \Omega_y \vec{e}_y + \Omega_z \vec{e}_z$.
In 2D, there is only one component of vorticity, $\Omega_z$:
$\Omega_z = \frac{\partial v}{\partial x} - \frac{\partial u}{\partial y}$.\
This is like imagining the domain is a 2D sheet of paper, where the axis of rotation points directly into the paper, and rotation at each point in space and time in the domain is of a particular strength, and of either clockwise or anti-clockwise predisposition.

-----------------------------------------

###### Vorticity Transport Equations

Start with the incompressible Navier-Stokes,

$\frac{\partial \vec{u}}{\partial t} + (\vec{u} \cdot \nabla)\vec{u} = \vec{g} - \frac{1}{\rho} \nabla p + \nu \nabla^2 \vec{u}$

If we do some jiggery-pokery and take the curl, we eventually get the 3D Vorticity Transport Equation:

$\frac{\partial \vec{\Omega}}{\partial t} + (\vec{u} \cdot \nabla) \vec{\Omega} - (\vec{\Omega} \cdot \nabla)\vec{u} = \nu \nabla^2 \vec{\Omega}$

Here, the first term is unsteady variation, the second is non-linear convection, the third is vortex stretching, the last is vortex diffusion. Simplifying slightly with the material derivative:

$\frac{D \vec{\Omega}}{Dt} = \nu \nabla^2 \vec{\Omega} + (\vec{\Omega} \cdot \nabla) \vec{u}$

Now in 2D, the last term, the vortex stretching term, disappears.

$\frac{\partial \vec{\Omega}}{\partial t} + (\vec{u} \cdot \nabla) \vec{\Omega} = \nu \nabla^2 \vec{\Omega}$

And, there's only one component of vorticity, $\Omega_z$:

$\boxed{\frac{\partial \Omega_z}{\partial t} + (\vec{u} \cdot \nabla) \Omega_z = \nu \nabla^2 \Omega_z}$

In scalar form,

$\boxed{\frac{\partial \Omega_z}{\partial t} + u \frac{\partial \Omega_z}{\partial x} + v \frac{\partial \Omega_z}{\partial y}= \nu \left(\frac{\partial^2 \Omega_z}{\partial x^2} + \frac{\partial^2 \Omega_z}{\partial y^2}\right)}$

Then, discretise the above equation using a suitable finite differencing method.

-------------------------------

###### Streamfunction-Poisson Equation

$u = \frac{\partial \psi}{\partial y} ; v = - \frac{\partial \psi}{\partial x}$

$\Omega_z = \frac{\partial v}{\partial x} - \frac{\partial u}{\partial y} = -\frac{\partial}{\partial x}\left(\frac{\partial \psi}{\partial x}\right) - \frac{\partial}{\partial y}\left(\frac{\partial \psi}{\partial y}\right) $

$\nabla^2 = -{\vec{\Omega}(x,y;t)}$

In scalar form,

$\boxed{\frac{\partial^2 \psi}{\partial x^2} + \frac{\partial^2 \psi}{\partial x^2} = -\Omega_z(x,y;t)}$

Discretise. Solve with sub-iterations for every cell in every time step.

#### Streamfunction-Vorticity Formulation Algorithm

1. Solve the _**discretised**_ streamfunction-Poisson equation for the streamfunction scalar field using initial vorticity values. Use sub-iterations and over-relaxation to solve it at every cell at every time step.
$\frac{\partial^2 \psi}{\partial x^2} + \frac{\partial^2 \psi}{\partial x^2} = -\Omega_z(x,y;t)$
2. Solve for velocities using the streamfunction scalar field:
$u = \frac{\partial \psi}{\partial y} ; v = - \frac{\partial \psi}{\partial x}$
3. Solve the _**discretised**_ 2D vorticity transport equation for the vorticity vector field:
$\frac{\partial \Omega_z}{\partial t} + u \frac{\partial \Omega_z}{\partial x} + v \frac{\partial \Omega_z}{\partial y}= \nu \left(\frac{\partial^2 \Omega_z}{\partial x^2} + \frac{\partial^2 \Omega_z}{\partial y^2}\right)$
4. Solve the pressure-Poisson for the pressure field.



