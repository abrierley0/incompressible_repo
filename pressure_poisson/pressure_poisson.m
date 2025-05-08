% Numerical Solution of the Incompressible Navier-Stokes
% Equations using the Pressure-Poisson Approach of
% Harlow and Welch (Los Alamos, 1951)

% FOR TWO-DIMENSIONAL CHANNEL FLOW

% Written by Mr Adam James Brierley
% 6th April 2025
% Cranfield University
% Centre for Computational Engineering Sciences (CES)
% Cranfield, Bedfordshire, MK43 0AL, UK

clc
clear all
close all
format long

%--------------------------------
% COMPUTATIONAL DOMAIN
%--------------------------------     
%                                       NORTH
%                                     dp/dy = 0
%                   i = 1, j = jmax                   i = imax, j = jmax
%                       D-------------------------------------------C
%                       |*******************************************|
%                       |*    INTERNAL DOMAIN                      *|
%                       |*    i = 2, ..., imax - 1                 *|
%        -dp/dx = 0     |*    j = 2, ..., jmax - 1                 *|    dp/dx = 0
%          WEST         |*                                         *|    EAST
%                       |*                                         *|
%                       |*******************************************|
%                       A-------------------------------------------B
%                   i = 1,                                i = imax,
%                   j = 1            -dp/dy = 0           j = jmax
%                                       SOUTH

%=========================
% INPUT VALUES
%=========================
L = 2;     % Length of channel [m]
H = 0.75;  % Height of channel [m]
W = 0.01;  % Width of channel  [m]

imax = 100;
jmax = 40;

% NOTE: remember to subtract 1 for the spaces
dx = L / (imax-1);
dy = H / (jmax-1);

% Fluid properties - water
rho = 997;       % Density [kg/m^3]
mu = 1.0016;     % Dynamic viscosity [Pa*s]


%==================
% INITIALISE
%==================
X = zeros(imax,jmax);  % NOTE: X and Y are capitalised
Y = zeros(imax,jmax);

u = zeros(imax,imax);
v = zeros(imax,jmax);


%=======================
% GRID DEFINITION
%=======================
% NOTE: x and y are lower case
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


