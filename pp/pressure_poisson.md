Pressure-Poisson Solver
=======================

Method originally used by Harlow & Welch at Los Alamos in the 1950s.

The Pressure-Poisson equation is derived quite simply by taking the divergence of the 3D incompressible Navier-Stokes equations, and then using several vector identities.

<u>Unknowns (Primitive Variables)</u>

$$p = p(x,y,z;t) = ?$$ $$u = u(x,y,z;t) = ?$$ $$v = v(x,y,z;t) = ?$$ $$w = w(x,y,z;t) = ?$$

System in vector form:

$$\nabla \cdot \textbf{u} = 0$$

$$\frac{\partial \textbf{u}}{\partial t} + (\textbf{u} \cdot \nabla)\textbf{u} = \textbf{g} - \frac{1}{\rho} \nabla p + \nu \nabla^2 \textbf{u}$$

$$\nabla^2 p = - \rho \left[(\nabla \otimes \textbf{u}) \cdot \cdot (\nabla \otimes \textbf{u})\right]$$

System in scalar form:

$$ \frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} + \frac{\partial w}{\partial z} = 0$$

$$\frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} + v \frac{\partial u}{\partial y} + w \frac{\partial u}{\partial z} = g_x - \frac{1}{\rho} \frac{\partial p}{\partial x} + \nu \left(\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} + \frac{\partial^2 u}{\partial z^2} \right)$$

$$\frac{\partial v}{\partial t} + u \frac{\partial v}{\partial x} + v \frac{\partial v}{\partial y} + w \frac{\partial v}{\partial z} = g_y - \frac{1}{\rho} \frac{\partial p}{\partial y} + \nu \left(\frac{\partial^2 v}{\partial x^2} + \frac{\partial^2 v}{\partial y^2} + \frac{\partial^2 v}{\partial z^2} \right)$$

$$\frac{\partial w}{\partial t} + u \frac{\partial w}{\partial x} + v \frac{\partial w}{\partial y} + w \frac{\partial w}{\partial z} = g_z - \frac{1}{\rho} \frac{\partial p}{\partial z} + \nu \left(\frac{\partial^2 w}{\partial x^2} + \frac{\partial^2 w}{\partial y^2} + \frac{\partial^2 w}{\partial z^2} \right)$$

$$\frac{\partial^2 p}{\partial x^2} + \frac{\partial^2 p}{\partial y^2} + \frac{\partial^2 p}{\partial z^2} = - \rho \left[ \left(\frac{\partial u}{\partial x} \right)^2 + \left( \frac{\partial v}{\partial y} \right)^2 + \left( \frac{\partial w}{\partial z} \right)^2 + 2 \left(\frac{\partial v}{\partial x} \cdot \frac{\partial v}{\partial x} + \frac{\partial v}{\partial x} + \frac{\partial v}{\partial x} \right) \right]$$

To start the iterative process we require an intial velocity field everywhere and an intial pressure field everywhere.

<u>Intial Conditions</u>

$$\textbf{u} = \textbf{u}_0 (\textbf{r}; t=0)$$
$$p = p_0(\textbf{r};t=0)$$

and these must satisfy the divergence-free constraint.
