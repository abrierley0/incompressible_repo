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
	REAL :: dx  ! Grid spacing in x [m]
	REAL :: dy  ! Grid spacing in y [m]
	REAL :: i   ! x counter
	REAL :: j   ! y counter
	REAL :: xx  ! x distance
	REAL :: yy  ! y distance

	! Constant Reals
	REAL, PARAMETER :: L = 0.1  ! Channel length [m]
	REAL, PARAMETER :: W = 1.0  ! Channel width [m]
	REAL, PARAMETER :: H = 0.02 ! Channel height [m]
	REAL, PARAMETER :: BETA = 10.0    ! AC parameter
	REAL, PARAMETER :: MU = 0.001003  ! Dynamic viscosity [Pa*s]
	REAL, PARAMETER :: RHO = 998.2    ! Density of water [kg/m^3]
	REAL, PARAMETER :: D_TAU = 0.001  ! Pseudo-time-step [s]

	! Declare arrays
	REAL, ALLOCATABLE :: x(:,:)   ! Current x [m]
	REAL, ALLOCATABLE :: y(:,:)   ! Current y [m]   
	REAL, ALLOCATABLE :: un(:,:)  ! Current u [m/s]
	REAL, ALLOCATABLE :: vn(:,:)  ! Current v [m/s]
	REAL, ALLOCATABLE :: pn(:,:)  ! Current p [Pa]
	REAL, ALLOCATABLE :: u(:,:)   ! Next u [m/s]
	REAL, ALLOCATABLE :: v(:,:)   ! Next v [m/s]
	REAL, ALLOCATABLE :: p(:,:)   ! Next p [Pa]	
	REAL, ALLOCATABLE :: uanalytical(:,:)  ! Analytical memory [m/s]

	! Define flow quantities
	nu = MU / RHO
	ua = (RE * nu) / (2.0 * H)  
	dx = L / (IMAX - 1)
	dy = H / (JMAX - 1)
	dp = (12.0 * MU * L / H * H) * UA


	!=============================
	! COMPUTATIONAL DOMAIN SKETCH 
	!=============================

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


	!=================
	! ALLOCATE MEMORY 
	!=================
	ALLOCATE(x(IMAX,JMAX))
	ALLOCATE(y(IMAX,JMAX))

	ALLOCATE(un(IMAX,JMAX))
	ALLOCATE(vn(IMAX,JMAX))
	ALLOCATE(pn(IMAX,JMAX))

	ALLOCATE(u(IMAX,JMAX))
	ALLOCATE(v(IMAX,JMAX))
	ALLOCATE(p(IMAX,JMAX))

	ALLOCATE(uanalytical(IMAX,JMAX))

	!========================
	! SPATIAL DISCRETISATION 
	!========================

	xx = 0.0
	DO i = 1,IMAX
		yy = 0.0
		DO j = 1,JMAX
			X(i,j) = xx
			Y(i,j) = yy
			yy = yy + dy
		END DO
		xx = xx + dx
	END DO

	!WRITE(*,*) 'X: ', X
	!WRITE(*,*) 'Y: ', Y

	!=======================
	! BOUNDARY CONDITIONS 
	!=======================

	! North boundary
	j = JMAX
	DO i = 2, (IMAX - 1)
		u(i,j) = 0.0
		v(i,j) = 0.0
	END DO
	! South boundary
	j = 1
	DO i = 2, (IMAX - 1)
		u(i,j) = 0.0
		v(i,j) = 0.0
	END DO
	! West boundary
	i = 1
	DO j = 2, (JMAX - 1)
		u(i,j) = ua
		v(i,j) = 0.0
	END DO

	!WRITE(*,*) 'u(i,j) = ', u
	!WRITE(*,*) 'v(i,j) = ', v

	!========================================
	! INITIAL CONDITION (ON INTERNAL DOMAIN)
	!========================================
	
	DO i = 2, (IMAX - 1)
		DO j = 2, (JMAX - 1)
			un(i,j) = ua
			vn(i,j) = 0.0
			pn(i,j) = 0.0
		END DO
	END DO

	!=======================
	! ANALYTICAL SOLUTION
	!=======================

	DO i = 1, IMAX
		DO j = 1, JMAX
		uanalytical(i,j) = (dp / 2.0 * MU * L) * y(i,j) * (H - y(i,j))
		END DO
	END DO

	!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	!===================
	! START MAIN LOOP
	!===================

	DO 1, NMAX

	! Assign fields of new variables to old

	! 1. Solve perturbed Continuity equation on internal domain
	! Update pressure boundary conds on walls
	! Pressure at corner points

	! 2. Solve Navier-Stokes momentum equations
	! Update east boundary condition

	! Compute residuals
	! Display iterations and residuals

	END DO

	!=================
	! END MAIN LOOP
	!=================

	! Plots

			
	

	

END PROGRAM ac




































