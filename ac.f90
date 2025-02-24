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
	INTEGER, PARAMETER :: NMAX = 40000  ! Number of iterations

	! Reals
	REAL :: ua  ! Inlet velocity [m/s]
	REAL :: nu  ! Kinematic viscosity [m^2/s]
	REAL :: dp  ! Pressure differential for analytical solution [Pa]

	! Constant Reals
	REAL, PARAMETER :: L = 0.1  ! Channel length [m]
	REAL, PARAMETER :: W = 1.0  ! Channel width [m]
	REAL, PARAMETER :: H = 0.02 ! Channel height [m]
	REAL, PARAMETER :: BETA = 10.0    ! AC parameter
	REAL, PARAMETER :: MU = 0.001003  ! Dynamic viscosity [Pa*s]
	REAL, PARAMETER :: RHO = 998.2    ! Density of water [kg/m^3]
	REAL, PARAMETER :: D_TAU = 0.001  ! Pseudo-time-step [s]

	! Define flow quantities

	ua = (RE * NU) / (2.0 * H)  
	nu = MU / RHO
	dx = L / (IMAX - 1)
	dy = H / (JMAX - 1)
	dp = (12.0 * MU * L / H * H) * UA


	!=============================!
	! COMPUTATIONAL DOMAIN SKETCH !
	!=============================!

	!            (i = 1,          dp/dy = 0        (i = imax,
	!             j = jmax)                         j = jmax)
	!               /-----------------------------------/
	!               /***********************************/
	!               /* Solution is on the internal     */
	!               /* domain, i = 2,...,imax          */
	!  -dp/dx = 0   /*         j = 2,...,jmax .        */   dp/dx = 0
	!               /*                                 */
	!               /*                                 */
	!               /*                                 */
	!               /*                                 */
	!               /***********************************/
	!               /-----------------------------------/
	!             (i = 1,                             (i = imax,
	!              j = 1)         -dp/dy = 0           j = 1)


	!=================!
	! ALLOCATE MEMORY !
	!=================!

	! Allocatable Reals
	REAL, ALLOCATABLE :: X(:,:)   ! Current x [m]
	REAL, ALLOCATABLE :: Y(:,:)   ! Current y [m]   

	REAL, ALLOCATABLE :: un(:,:)  ! Current u [m/s]
	REAL, ALLOCATABLE :: vn(:,:)  ! Current v [m/s]
	REAL, ALLOCATABLE :: pn(:,:)  ! Current p [Pa]

	REAL, ALLOCATABLE :: u(:,:)   ! Next u [m/s]
	REAL, ALLOCATABLE :: v(:,:)   ! Next v [m/s]
	REAL, ALLOCATABLE :: p(:,:)   ! Next p [Pa]	

	REAL, ALLOCATABLE :: unanalytical(:,:)  ! Analytical memory [m/s]



END PROGRAM ac
