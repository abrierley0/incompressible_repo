Pressure-Poisson Solver
=======================

Method originally used by Harlow & Welch at Los Alamos in the 1950s.

The Pressure-Poisson equation is derived quite simply by taking the divergence of the 3D incompressible Navier-Stokes equations, and then using several vector identities.

System in vector form:

$$\nabla \cdot \textbf{u} = 0$$

$$\nabla^2 p = - \rho \left[(\nabla \otimes \textbf{u}) \cdot \cdot (\nabla \otimes \textbf{u})\right]$$

System in scalar form:

$$ \frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} + \frac{\partial w}{\partial z} = 0$$

$$\frac{\partial^2 p}{\partial x^2} + \frac{\partial^2 p}{\partial y^2} + \frac{\partial^2 p}{\partial z^2} = - \rho \left[ \left(\frac{\partial u}{\partial x} \right)^2 + \left( \frac{\partial v}{\partial y} \right)^2 + \left( \frac{\partial w}{\partial z} \right)^2 + 2 \left(\frac{\partial v}{\partial x} \cdot \frac{\partial v}{\partial x} + \frac{\partial v}{\partial x} + \frac{\partial v}{\partial x} \right) \right]$$


