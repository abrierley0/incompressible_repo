Pressure-Poisson Solver
=======================

The method of Harlow & Welch, Los Alamos (1951). This was one of the earliest solvers and one of the most efficient. The essence is to take the divergence of the incompressible Navier-Stokes equation, resulting in cancellation of unsteady, gravity, and viscous terms, leaving

$$\nabla \cdot (\mathbf{u} \cdot \nabla) \mathbf{u} = - \frac{1}{\rho} \nabla \cdot \nabla p $$

Then there are two vector identities used to change the LHS and you multiply by $- \rho$. The result is the double-dot scalar product of two second-rank tensors (the velocity-gradient tensors),

$$ \nabla ^2 p = - \rho \left[(\nabla \otimes \mathbf{u}) \cdot \cdot (\nabla \otimes \mathbf{u})\right]$$

Expanding for the scalar form of the Pressure-Poisson equation,

$$ \frac{\partial^2 p}{\partial x^2} + \frac{\partial^2 p}{\partial y^2} + \frac{\partial^2 p}{\partial z^2} = \rho \left[ \left(\frac{\partial u}{\partial x}\right)^2 + \left(\frac{\partial v}{\partial y}\right)^2 + \left(\frac{\partial w}{\partial z}\right)^2 + 2 \left(\frac{\partial v}{\partial x} \cdot \frac{\partial u}{\partial y} + \frac{\partial w}{\partial x} \cdot \frac{\partial u}{\partial z} + \frac{\partial w}{\partial y} \cdot \frac{\partial v}{\partial z}\right) \right]$$

We need a numerical solution for this equation!!

For simplicity's sake, reduce the problem to two dimensions.

First we need to develop an algorithm to solve.

