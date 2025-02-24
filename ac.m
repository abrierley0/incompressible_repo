% A.C Method Implementation in Matlab
% For Incompressible Channel Flows
% Method is formulated by Chorin in 1967

% Written by Mr A. J. Brierley
% Centre for Computational Engineering Sciences (CES)
% Cranfield University
% Bedfordshire, MK43 0AL, UK
% 20/02/2025

clear all; close all; clc;
format long;

%============================%
% INPUT DATA FOR SIMULATIONS %
%============================%

Re = 100.0;        % Reynolds number [-]
mu = 0.001003;     % Dynamic viscosity of water 20C [Pa*s]
rho = 998.2;       % Density of water [kg/m^3]
nu = mu / rho;      % Kinematic viscosity [m^2/s]

L = 0.1;    % Length of channel [m]
W = 1.0;    % Width of channel [m]
H = 0.02;   % Height of channel [m]

imax = 100;  % Max cells in the x direction
jmax = 41;  % Max cells in the y direction

dx = L / (imax - 1);   % Grid spacing in x
dy = H / (jmax -1);    % Grid spacing in y

beta = 10.0;  % A.C. Parameter

ua = Re * nu * (1 / (2.0 * H));  % Average velocity [m/s]
dp = (12.0 * mu * L / H^2) * ua;  % Pressure Differential [Pa]

nmax = 40000;  % Max. number of time steps
d_tau = 0.001;  % Pseudo- time step [s]


%===============%
% DOMAIN SKETCH %
%===============%

%           (i = 1,        dp/dy = 0          (i = imax,
%           j - jmax)         NORTH            j = jmax)
%              -------------------------------------
%              /***********************************/
%              /* Solution is in the internal     */
%              /* domain                          */
%              /* Internal domain is              */
% -dp/dx = 0   /* i = 2,...,imax - 1              */   dp/dx = 0
%    WEST      /* j = 2,...,jmax - 1              */    EAST
%              /*                                 */
%              /* NOT TO SCALE                    */
%              /***********************************/
%              -------------------------------------
%           (i = 1,        SOUTH              (i = imax,
%           j = 1)      -dp/dy = 0             j = jmax)

%===================%
% INITIALISE ARRAYS %
%===================%

X = zeros(imax,jmax);
Y = zeros(imax,jmax);

% Note the stupid notation, u, v, and p, are new values
u = zeros(imax,jmax);
v = zeros(imax,jmax);
p = zeros(imax,jmax);

% 'un' is denoting u^(n), that is the current time level
un = zeros(imax,jmax);
vn = zeros(imax,jmax);
pn = zeros(imax,jmax);

unanalytical = zeros(imax,jmax);

%========================%
% SPATIAL DISCRETISATION %
%========================%

x = 0.0;
for i = 1:imax
    y = 0.0;
    for j = 1:jmax
        X(i,j) = x;
        Y(i,j) = y;
        y = y + dy;
    end
    x = x + dx;
end

% Note: the final + dx isn't stored in the array,
% so x goes past the domain but it isn't stored.

%============================================%
% BOUNDARY CONDITIONS AND INITIAL CONDITIONS %
%============================================%

% NORTH WALL 
j = jmax;
for i = 2:(imax-1)
    u(i,j) = 0.0;
    v(i,j) = 0.0;
end

% WEST WALL
i = 1;
for j = 2:(jmax-1)
    u(i,j) = ua;
    v(i,j) = 0.0;
end

% SOUTH WALL
j = 1;
for i = 2:(imax-1)
    u(i,j) = 0.0;
    v(i,j) = 0.0;
end

% INITIAL CONDITION
for i = 2:(imax-1)
    for j = 2:(jmax-1)
        un(i,j) = ua;
        vn(i,j) = 0.0;
        pn(i,j) = 0.0;
    end
end

%=====================%
% ANALYTICAL SOLUTION %
%=====================%

for i = 1:imax
    for j = 1:jmax
        uanalytical(i,j) = (dp / 2.0 * mu * L) * Y(i,j) * (H - Y(i,j));
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%=================%
% START MAIN LOOP %
%=================%

for n = 1:nmax

% ASSIGN FIELDS OF NEW VALUES ONTO OLD ONES
for i = 1:(imax)
    for j = 1:(jmax)
        un(i,j) = u(i,j);
        vn(i,j) = v(i,j);
        pn(i,j) = p(i,j);
    end
end

%==================================================%
% 1. SOLVE PERTURBED CONTINUITY ON INTERNAL DOMAIN %
%==================================================%

for i = 2:(imax-1)
    for j = 2:(jmax-1)
        RHSP = ((1/dx)*(un(i,j) - un(i-1,j)) + (1/dy)*(vn(i,j) - vn(i,j-1)));
        p(i,j) = pn(i,j) - beta * d_tau * RHSP;
    end
end

% UPDATE PRESSURE B.C.S ON WALLS
% INLET (WEST) AND OUTLET (EAST) B.C.S UPDATE
for j = 2:(jmax-1)
    p(1,j) = p(2,j);
    p(imax,j) = 2.0*p((imax-1),j) - p(imax-2,j);  % SECOND ORDER
end
% SOUTH AND NORTH BOUNDARIES
for i = 2:(imax-1)
    p(i,1) = p(i,2);
    p(i,jmax) = p(i,(jmax-1));
end

% PRESSURE AT CORNER POINTS
% TAKE AVERAGES OF NORMALLY ADJACENT CELLS
p(1,1) = (p(2,1) + p(1,2))/2.0;
p(imax,jmax) = (p(imax-1,jmax) + p(imax,jmax-1))/2.0;
p(imax,1) = (p(imax-1,1) + p(imax,2))/2.0;
p(1,jmax) = (p(2,jmax) + p(1,jmax-1))/2.0;

%=====================================%
% 2. SOLVE U AND V MOMENTUM EQUATIONS %
%=====================================%

for i = 2:(imax-1)
    for j = 2:(jmax-1)
        dpdx = (p(i+1,j) - p(i,j))/dx;
        dpdy = (p(i,j+1) - p(i,j))/dy;
        d2udx2 = (un(i+1,j) - 2.0*un(i,j) + un(i-1,j))/(dx*dx);
        d2udy2 = (un(i,j+1) - 2.0*un(i,j) + un(i,j-1))/(dy*dy);
        d2vdx2 = (vn(i+1,j) - 2.0*vn(i,j) + vn(i-1,j))/(dx*dx);
        d2vdy2 = (vn(i,j+1) - 2.0*vn(i,j) + vn(i,j-1))/(dy*dy);
        dudx = (un(i,j) - un(i-1,j))/dx;
        dudy = (un(i,j) - un(i,j-1))/dy;
        dvdx = (vn(i,j) - vn(i-1,j))/dx;
        dvdy = (vn(i,j) - vn(i,j-1))/dy;
        RHSU = -dpdx/rho + nu * (d2udx2 + d2udy2) - un(i,j)*dudx - vn(i,j)*dudy;
        RHSV = -dpdy/rho + nu * (d2vdx2 + d2vdy2) - un(i,j)*dvdx - vn(i,j)*dvdy;
        u(i,j) = un(i,j) + d_tau * RHSU;
        v(i,j) = vn(i,j) + d_tau * RHSV;
    end
end 

% UPDATE EAST (OUTLET) VELOCITY B.C.
i = imax;
for j = 2:(jmax-1)
    u(i,j) = u(i-1,j);
    v(i,j) = u(i-1,j);
end

%===================%
% COMPUTE RESIDUALS %
%===================%
sumU = 0.0;
sumV = 0.0;
sumP = 0.0;

for i = 2:(imax-1)
    for j = 2:(jmax-1)
        sumU = sumU + abs(u(i,j) - un(i,j));
        sumV = sumV + abs(v(i,j) - vn(i,j));
        sumP = sumP + abs(p(i,j) - pn(i,j));
    end
end

clc;
display("AC method of Chorin incompressible Navier-Stokes solver:")
sumU
sumV
sumP
display("Number of iterations:")
n

end
%===============%
% END MAIN LOOP %
%===============%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%=======%
% PLOTS %
%=======%

figure(1);
colormap(jet);
pcolor(X,Y, u);
title('Outlet Velocity Profile','fontsize',16);
xlabel('y[m]','fontsize',16);
ylabel('x[m]','fontsize',16);
grid on;
disp('PRESS A BUTTON!');
pause;
close(1);

figure(2);
plot(Y(imax,:),uanalytical(imax,:),'r');
hold on;
plot(Y(imax,:),u(imax,:),'bo');
hold off;
legend('Analytical Solution','Numerical Solution (AC)');
title('Outlet Velocity Profile','fontsize',16);
xlabel('y[m]','fontsize',16);
ylabel('u(y) [m/s]','fontsize',16);
grid on;
disp('PRESS A BUTTON!');
pause;
close(2);
