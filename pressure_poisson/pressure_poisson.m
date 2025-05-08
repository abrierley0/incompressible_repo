% Numerical Solution of the Incompressible Navier-Stokes
% Equations using the Pressure-Poisson Approach of
% Harlow and Welch (Los Alamos, 1951)

% FOR TWO-DIMENSIONAL CHANNEL FLOW

% Written by Mr Adam James Brierley
% 6th April 2025
% Cranfield University
% Centre for Computational Engineering Sciences (CES)
% Cranfield, Bedfordshire, MK43 0AL, UK

% Written in MATLAB programming language

clc
clear all
close all
format long

%=============================
% COMPUTATIONAL DOMAIN
%============================   
%                                       NORTH
%                                     dp/dy = 0
%                   (i = 1, j = jmax)                   (i = imax, j = jmax)
%                       D-------------------------------------------C
%                       |*******************************************|
%                       |*    INTERNAL DOMAIN                      *|
%                       |*    i = 2, ..., imax - 1                 *|
%        -dp/dx = 0     |*    j = 2, ..., jmax - 1                 *|    dp/dx = 0
%          WEST         |*                                         *|    EAST
%                       |*                                         *|
%                       |*******************************************|
%                       A-------------------------------------------B
%                   (i = 1,                                (i = imax,
%                   j = 1)            -dp/dy = 0           j = jmax)
%                                       SOUTH

%=========================
% INPUT VALUES
%=========================
Re = 100   % Reynolds number

L = 2;     % Length of channel [m]
H = 0.75;  % Height of channel [m]
W = 0.01;  % Width of channel  [m]

imax = 100;
jmax = 40;
nmax = 40000;  % Max. time steps

dx = L / (imax-1);  % NOTE: remember to subtract 1 for the spaces
dy = H / (jmax-1);

rho = 997;       % Density [kg/m^3]
mu = 1.0016;     % Dynamic viscosity [Pa*s]
nu = mu / rho;

ua = 1;  % Initial velocity [m/s]


%==================
% INITIALISE
%==================
X = zeros(imax,jmax);  % NOTE: X and Y are uppercase
Y = zeros(imax,jmax);

u = zeros(imax,jmax);  % New
v = zeros(imax,jmax);
p = zeros(imax,jmax);

un = zeros(imax,jmax);  % Old
vn = zeros(imax,jmax);
pn = zeros(imax,jmax);


%=======================
% GRID DEFINITION
%=======================
% NOTE: x and y are lowercase
y = 0.0;
for j = 1:jmax
    x = 0.0;
    for i = 1:imax
        X(i,j) = x;
        Y(i,j) = y;
        x = x + dx;  % NOTE: last one isn't recorded
    end
    y = y + dy;
end

%==================================
% INITIAL AND BOUNDARY CONDITIONS
%==================================
% INITIAL CONDITIONS
for i = 1:imax
    for j = 2:jmax-1
        u(i,j) = ua;
        v(i,j) = 0.0;
        p(i,j) = 0.0;
    end 
end

% WEST (INLET) BOUNDARY CONDITION
for i = 1
    for j = 2:(jmax-1)
        u(i,j) = ua;
        v(i,j) = 0.0;
    end
end

% EAST (OUTLET) BOUNDARY CONDITION
for i = imax
    for j = 2:(jmax-1)
        u(i,j) = 0.0;
        v(i,j) = 0.0;
    end
end

% NORTH (UPPER WALL) BOUNDARY CONDITION
for j = jmax
    for i = 2:(imax-1)
        u(i,j) = 0.0;
        v(i,j) = 0.0;
    end
end

% SOUTH (LOWER WALL) BOUNDARY CONDITION
for j = 1
    for i = 2:(imax-1)
        u(i,j) = 0.0;
        v(i,j) = 0.0;
    end
end


%============================
% START MAIN TIME LOOP
%============================

for n = 1:nmax

    % ASSIGN NEW VALUES TO THE OLD VALUES
    for i = 1:imax
        for j = 1:jmax
            un(i,j) = u(i,j);
            vn(i,j) = v(i,j);
            pn(i,j) = p(i,j);
        end
    end

    %==============================================
    % SOLVE THE NAVIER-STOKES MOMENTUM EQUATIONS
    %==============================================

    % Compute viscous terms with central diff.
    d2udx2 = (un(i+1,j)-2*un(i,j)+un(i-1,j))/(dx*dx);
    d2udy2 = (un(i,j+1)-2*un(i,j)+un(i,j-1))/(dy*dy);
    LaplacianU = d2udx2 + d2udy2;

    d2vdx2 = (vn(i+1,j)-2*vn(i,j)+vn(i-1,j))/(dx*dx);
    d2vdy2 = (vn(i,j+1)-2*vn(i,j)+vn(i,j-1))/(dy*dy);
    LaplacianV = d2vdx2 + d2vdy2;

    % Compute 



end