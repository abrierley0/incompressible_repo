% Numerical Solution of the Navier-Stokes
% Equations Using the Fractional-Step Pressure Projection
% Formulation of Chorin, 1968

% Written by Dr. Laszlo Konozsy ' 29. 01. 2025
% Cranfield University,
% Faculty of Engineering and Applied Sciences,
% Cranfield, Bedfordshire, MK43 0AL, UK

% clear the screen
clc; 

% clear all variables
clear all;

% use double-precision
format long;

%=================================
% INPUT DATA FOR THE SIMULATIONS
%=================================

Re = 100.0;

L = 1.0;  % Length of the channel [m]
h = 0.02; % Height of the channel [m]
w = 1.0;  % Width of the channel [m]

imax = 50;
jmax = 41;
nmaxNS = 4000;
itPPMAX = 100;
omega = 1.7;
dt = 0.01;

rho = 998.2;  % Density of the fluid [kg/m^3]
mu = 0.001003; % Dynamic viscosity of the fluid [Pa*s]
nu = mu/rho;  % Kinematic viscosity of the fluid [m^2/s]

ua = (Re*nu)/(2.0*h);
dp = (12.0*mu*L*ua)/(h*h);

%                         (i = 1,          dp/dy = 0        (i = imax,
%                          j = jmax)       NORTH             j = jmax)
%                          D -------------------------------- C
%                          | ******************************** |
%                          | * Solution is in the internal  * |
%                          | * domain                       * |
%                          | * i = 2,...,(imax-1)           * |
%                          | * j = 2,...,(jmax-1)           * |
% -dp/dx = 0               | *                              * |    dp/dx = 0
%  WEST                    | *                              * |    EAST
%                          | *                              * |
%                          | *                              * |
%                          | *                              * |
%                          | ******************************** |
%                          A -------------------------------- B
%                         (i = 1,         SOUTH               (i = imax,
%                          j = 1)         -dp/dy = 0          j = 1)


X = zeros(imax,jmax);
Y = zeros(imax,jmax);

u = zeros(imax,jmax);
v = zeros(imax,jmax);
p = zeros(imax,jmax);

ustar = zeros(imax,jmax);
vstar = zeros(imax,jmax);
RHSP = zeros(imax,jmax);


un = zeros(imax,jmax);
vn = zeros(imax,jmax);
pn = zeros(imax,jmax);

uanalytical = zeros(imax,jmax);

%==============================
% Rectangular Mesh Generation
%==============================

dx = L/(imax-1);
dy = h/(jmax-1);

y = 0.0;
for j = 1:jmax
 x = 0.0;
 for i = 1:imax
  X(i,j) = x;
  Y(i,j) = y;
  x = x + dx;
 end
 y = y + dy;
end 

% SIMPLIFIED ANALYTICAL SOLUTION OF THE NAVIER-STOKES EQUATIONS FOR CHANNELS
for i = 1:imax
 for j = 1:jmax
  uanalytical(i,j) = (dp/(2.0*mu*L))*Y(i,j)*(h-Y(i,j));
 end
end

% WEST (INLET) BOUNDARY CONDITIONS
i = 1;
for j = 2:(jmax-1)
 u(i,j) = ua;
 v(i,j) = 0.0;
end

% NORTH (UPPER WALL) BOUNDARY CONDITIONS
j = jmax;
for i = 2:(imax-1)
 u(i,j) = 0.0;
 v(i,j) = 0.0;
end

% SOUTH (LOWER WALL) BOUNDARY CONDITIONS
j = 1;
for i = 2:(imax-1)
 u(i,j) = 0.0;
 v(i,j) = 0.0;
end

% INITIAL CONDITIONS
for j = 2:(jmax-1)
 for i = 2:(imax-1)
  un(i,j) = ua; 
  vn(i,j) = 0.0; 
  pn(i,j) = 0.0; 
 end
end 

% MAIN LOOP
for n=1:nmaxNS

 % COPYING THE FIELDS OF NEW VALUES TO THE OLD VALUES
 for j = 1:jmax
  for i = 1:imax
   un(i,j) = u(i,j); 
   vn(i,j) = v(i,j); 
   pn(i,j) = p(i,j); 
   ustar(i,j) = u(i,j); 
   vstar(i,j) = v(i,j); 
  end
 end 
 
%=============================================%
% BOUNDARY CONDITIONS AND THEIR SPECIFICATION %
%=============================================%

%====================================================================================%
% 1. FRACTIONAL-STEP: COMPUTATION OF THE INTERMEDIATE VELOCITY FIELDS (USTAR, VSTAR) %
%====================================================================================%

 for j = 2:(jmax-1)
  for i = 2:(imax-1)
   % Components of the gravity field vector
   gx = 0.0;
   gy = 0.0;

   % Compute the viscous terms
   d2undx2 = (un(i+1,j)-2.0*un(i,j)+un(i-1,j))/(dx*dx);
   d2undy2 = (un(i,j+1)-2.0*un(i,j)+un(i,j-1))/(dy*dy);
   LaplacianU = d2undx2+d2undy2;
  
   d2vndx2 = (vn(i+1,j)-2.0*vn(i,j)+vn(i-1,j))/(dx*dx);
   d2vndy2 = (vn(i,j+1)-2.0*vn(i,j)+vn(i,j-1))/(dy*dy);
   LaplacianV = d2vndx2+d2vndy2;
  
   % Compute the convective terms
   dundx = (un(i,j)-un(i-1,j))/dx; 
   dundy = (un(i,j)-un(i,j-1))/dy;

   dvndx = (vn(i,j)-vn(i-1,j))/dx; 
   dvndy = (vn(i,j)-vn(i,j-1))/dy;
   
   % Compute the velocity field in the internal domain
   ustar(i,j) = un(i,j)+dt*(gx+nu*LaplacianU-un(i,j)*dundx-vn(i,j)*dundy);
   vstar(i,j) = vn(i,j)+dt*(gy+nu*LaplacianV-un(i,j)*dvndx-vn(i,j)*dvndy);
  end 
 end 

 %==============================================================================%
 % RIGHT HAND SIDE OF THE PRESSURE-POISSON EQUATION: (rho/dt)*DIV(USTAR_VECTOR) %
 %==============================================================================%
 for j = 2:(jmax-1)
  for i = 2:(imax-1)
   dustardx = (ustar(i,j)-ustar(i-1,j))/dx;
   dvstardy = (vstar(i,j)-vstar(i,j-1))/dy;
   RHSP(i,j) = (rho/dt)*(dustardx+dvstardy);
  end 
 end 
 
%===========================================================================%
% 2. SOLUTION OF THE PRESSURE-POISSIN EQUATION WITH THE POINT S.O.R. METHOD %
%===========================================================================%

 gamma = (dx/dy)*(dx/dy);
 beta = 2.0*(1.0+gamma);

 for itPP = 1:itPPMAX

  for j = 2:(jmax-1)
   for i = 2:(imax-1)   
    p(i,j) = (1.0-omega)*p(i,j)...
              +(omega/beta)*(p(i-1,j)+p(i+1,j)...
	  		  +gamma*(p(i,j-1)+p(i,j+1))-(dx*dx)*RHSP(i,j));
   end % end of "i" loop
  end % end of "j" loop
  
  %===============================================================
  % SPECIFICATION (UPDATE) OF THE PRESSURE BOUNDARY CONDITIONS
  %===============================================================
  % INLET (WEST) AND OUTLET (EAST) BOUNDARY CONDITIONS UPDATE
  for j = 2:(jmax-1)
   p(1,j) = p(2,j); 
   p(imax,j) = 2.0*p(imax-1,j)-p(imax-2,j);
   % p(imax,j) = 0.0; % DIRICHLET OUTLET BOUNDARY CONDITION
  end % end of "j" loop
  % LOWER (SOUTH) AND UPPER (NORTH) WALL BOUNDARY CONDITIONS UPDATE
  for i = 2:(imax-1)
   p(i,1) = p(i,2); 
   p(i,jmax) = p(i,jmax-1); 
  end   
  
  % PRESSURE AT THE CORNER POINTS A, B, C, D
  p(1,1) = (p(2,1)+p(1,2))/2.0;  
  p(imax,1) = (p(imax-1,1)+p(imax,2))/2.0;
  p(imax,jmax) = (p(imax,jmax-1)+p(imax-1,jmax))/2.0;
  p(1,jmax) = (p(2,jmax)+p(1,jmax-1))/2.0;
  
 end % END OF THE SUB-ITERATIONS FOR THE PRESSURE-POISSON EQ.
 
%================================================%
% 3. DIVERGENCE-FREE VELOCITY FIELD UPDATE (U,V) %
%================================================%

 for j = 2:(jmax-1)
  for i = 2:(imax-1)
   % Compute the pressure gradient terms
   dpdx = (p(i+1,j)-p(i,j))/dx;
   dpdy = (p(i,j+1)-p(i,j))/dy;
   % Compute the velocity field in the internal domain
   u(i,j) = ustar(i,j)-(dt/rho)*dpdx;   
   v(i,j) = vstar(i,j)-(dt/rho)*dpdy;
  end 
 end 

 % UPDATE EAST (OUTLET) VELOCITY BOUNDARY CONDITIONS
 i = imax;
 for j = 2:(jmax-1)
%  u(i,j) = u(i-1,j);
%  v(i,j) = v(i-1,j);
  u(i,j) = 2.0*u(i-1,j)-u(i-2,j);
  v(i,j) = 2.0*v(i-1,j)-v(i-2,j);
 end

 % COMPUTE THE RESIDUALS
 sumU = 0.0;
 sumV = 0.0;
 sumP = 0.0;

 for j = 2:(jmax-1)
  for i = 2:(imax-1)
   sumU = sumU + abs(u(i,j)-un(i,j));
   sumV = sumV + abs(v(i,j)-vn(i,j));
   sumP = sumP + abs(p(i,j)-pn(i,j));
  end 
 end 
  
 clc;
 disp('Navier-Stokes Solver with the FS-PP Method of Chorin in a 2D Channel (Written by Dr. Laszlo Konozsy):');
 sumU
 sumV
 sumP
 disp('Number of Iterations:');
 n

end % end of loop "n"

% COLOR PLOT OF THE VELOCITY COMPONENT-U
figure(1);
colormap(jet);
pcolor(X,Y,u);
title('Velocity Component U','fontsize',16);
xlabel('x[m]','fontsize',16);
ylabel('y[m]','fontsize',16);
disp('PRESS A BUTTON!');
pause;
close(1);

% COMPARISON OF THE ANALYTICAL SOLUTION WITH THE NUMERICAL SOLUTION
figure(2);
plot(Y(imax,:),uanalytical(imax,:),'r');
hold on;
plot(Y(imax,:),u(imax,:),'bo');
hold off;
legend('Analytical Solution','Numerical Solution (FS-PP)');
title('Outlet Velocity Profile','fontsize',16);
xlabel('y[m]','fontsize',16);
ylabel('u(y) [m/s]','fontsize',16);
grid on;
disp('PRESS A BUTTON!');
pause;
close(2);

% END OF THE CODE
