Pressure-Poisson Solver
=======================

This is the method of Harlow & Welch, Los Alamos (1951). This was one of the earliest solvers and one of the most efficient. The essential question is how can we solve the incompressible Navier-Stokes if we do not know the pressure field? The essence is to take the divergence of the incompressible Navier-Stokes equation, resulting in cancellation of unsteady, gravity, and viscous terms, leaving

$$\nabla \cdot (\mathbf{u} \cdot \nabla) \mathbf{u} = - \frac{1}{\rho} \nabla \cdot \nabla p $$

Then there are two vector identities used to change the LHS and you multiply by $- \rho$. The result is the double-dot scalar product of two second-rank tensors (the velocity-gradient tensors),

$$ \nabla ^2 p = - \rho \left[(\nabla \otimes \mathbf{u}) \cdot \cdot (\nabla \otimes \mathbf{u})\right]$$

Giving us an equation for the pressure. It can be seen that there is a strong dependence on the velocity-gradient tensors. **Note**: exactly why this equation can be used should be thought about.

Expanding for the scalar form of the Pressure-Poisson equation,

$$ \frac{\partial^2 p}{\partial x^2} + \frac{\partial^2 p}{\partial y^2} + \frac{\partial^2 p}{\partial z^2} = \rho \left[ \left(\frac{\partial u}{\partial x}\right)^2 + \left(\frac{\partial v}{\partial y}\right)^2 + \left(\frac{\partial w}{\partial z}\right)^2 + 2 \left(\frac{\partial v}{\partial x} \cdot \frac{\partial u}{\partial y} + \frac{\partial w}{\partial x} \cdot \frac{\partial u}{\partial z} + \frac{\partial w}{\partial y} \cdot \frac{\partial v}{\partial z}\right) \right]$$

We need a numerical solution for this equation!! There is then an algorithm with five equations involved - the continuity equation, the x-Navier-Stokes, y-Navier-Stokes, z- Navier-Stokes, and the above Pressure-Poisson. An algorithm must be developed to solve these. You can use the Gauss-Seidel Successive Over-Relaxation approach or also known as the point SOR approach. For the derivative terms, one can use appropriate finite differencing. A five-point discretisation stencil is used with this.

Let's write the system of equations in scalar form for two dimensions.

1. $$\frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} = 0$$

2. $$\frac{\partial u}{\partial t} + \left(u \frac{\partial u}{\partial x} + v \frac{\partial u}{\partial y}\right) = g_x - \frac{1}{\rho} \frac{\partial p}{\partial x} + \nu \left(\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2}\right)$$

3. $$\frac{\partial v}{\partial t} + \left(u \frac{\partial v}{\partial x} + v \frac{\partial v}{\partial y}\right) = g_y - \frac{1}{\rho} \frac{\partial p}{\partial y} + \nu \left(\frac{\partial^2 v}{\partial x^2} + \frac{\partial^2 v}{\partial y^2}\right)$$

4. $$\frac{\partial^2 p}{\partial x^2} + \frac{\partial^2 p}{\partial y^2} = -\rho \left[\left(\frac{\partial u}{\partial x}\right)^2 +\left(\frac{\partial v}{\partial y}\right)^2 + 2 \left(\frac{\partial v}{\partial x} \cdot \frac{\partial u}{\partial y}\right) \right]$$

Questions
---------

- Why can this equation for the pressure be used?
- Is it an approximation?
- Why do the boundary conditions use (u,v,p) but the initial conditions (un,vn,pn)?

Notes
-----

- 