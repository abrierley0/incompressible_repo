$u = \partial \psi /\partial y$\
$v = - \partial \psi / \partial x$

Streamfunction BCs
==================
Top Wall
--------

$u_{i,j} = \frac{\psi_{i,j} - \psi_{i,j-1}}{\Delta y}$

$\psi_{i,j} = u_{i,j} * \Delta y + \psi_{i,j-1}$

If dimensionless u(i,j) = 1

$v_{i,j} = - (\frac{\psi_{i,j} - \psi_{i-1,j}}{\Delta x}) = 0$

$\psi_{i,j} = \psi_{i-1,j}$  
if v(i,j) = 0

implies psi(i,j) can be specified as Dirichlet-type BC

Left Wall
---------

$u_{i,j} = \frac{\psi_{i,j} - \psi_{i,j-1}}{\Delta y} = 0$

$\psi_{i,j} = \psi_{i-1,j}$

$v_{i,j} = - (\frac{\psi_{i+1,j} - \psi_{i,j}}{\Delta x})$ = 0

$ \psi_{i,j} = \psi_{i+1,j}$      % May work even better, converge better

Also Dirichlet condition applied.
Generaly, say psi = 0 on wall, but approximate by $ \psi_{i,j} = \psi_{i+1,j}$.\
COnsistent derivation of BCs above.\
Other authors would use backward on LH wall, like top wall.

$v_{i,j} = - (\frac{\psi_{i,j} - \psi_{i-1,j}}{\Delta x})$ = 0

$\psi_{i-1,j} = 0$

$v_{i,j} = -\psi_{i,j} / \Delta x = 0 $

$\psi_{i,j} = 0$

$\Omega = \frac{\partial v}{\partial x} - \frac{\partial u}{\partial y}$

VORTICITY BCs
=============
TOP WALL
--------

$\Omega_{i,j} = \frac{v_{i,j} - v_{i-1,j}}{ \Delta x} - \frac{u_{i,j} - u_{i,j-1}}{\Delta y} = - (\frac{u_{i,j} - u_{i,j-1}}{\Delta y})$

This is first order derivatives, first order accuracy.\
Can use higher order backward schemes for the boundary conditions.\
Solve vorticity with explicit scheme.\
Consistently use correct derivatives at each wall - procedure is similar for 3D.






