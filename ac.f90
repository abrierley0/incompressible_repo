! A.C. Method of Chorin (1965) for the Solution to the Incompressible 
! Channel Flow

! Written by Mr A. J. Brierley

! Cranfield University
! Centre for Computational Engineering Sciences (CES)
! Bedfordshire, MK43 OAL, UK

! 24.02.2025



PROGRAM ac

IMPLICIT NONE

!==========================
! INTIALISE FLOW VARIABLES
!==========================

! Constant Integers
INTEGER, PARAMETER :: RE = 100    ! Reynolds number 
INTEGER, PARAMETER :: IMAX = 100  ! Max. cells in x direction
INTEGER, PARAMETER :: JMAX = 41   ! Max. cells in y direction

! Reals
REAL :: ua  ! Inlet velocity [m/s]
REAL :: nu  ! Kinematic viscosity [m^2/s]

! Constant Reals
REAL, PARAMETER :: L = 0.1  ! Channel length [m]
REAL, PARAMETER :: W = 1.0  ! Channel width [m]
REAL, PARAMETER :: H = 0.02 ! Channel height [m]
REAL, PARAMETER :: BETA = 10.0    ! AC parameter
REAL, PARAMETER :: MU = 0.001003  ! Dynamic viscosity [Pa*s]
REAL, PARAMETER :: RHO = 998.2    ! Density of water [kg/m^3]

INTEGER, PARAMETER :: NMAX = 40000  ! NUMBER OF ITERATIONS

ua = (RE * NU) / (2.0 * H)  





END PROGRAM ac
