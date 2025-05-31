! TDMA SUBROUTINE
!
! Written by Mr A. J. Brierley and Grok
! from Pletcher Appendix A
!
! 09.05.2025
!
! Cranfield University
! Centre for Computational Engineering Sciences (CES)
! Bedfordshire, MK43 0AL, UK

program tdma
    implicit none
    integer, parameter :: n = 6
    real, dimension(n) :: DD, BB, AA, CC
    integer :: IL, IU, I
    
    ! Initialise arrays based on the example
    DD = [2.0, 3.0, 4.0, 5.0, 7.0, 8.0]        ! Main diagonal
    BB = [0.0, 1.0, 2.0, 9.0, 2.0, 3.0]        ! Subdiagonal (BB(1) unused)
    AA = [1.0, 1.0, 0.0, 1.0, 3.0, 8.0]        ! Superdiagonal (AA(3) unused)
    CC = [5.0, 8.0, 12.0, 2.0, 4.0, 9.0]       ! Right-hand side
    IL = 1
    IU = n
    
    ! Call the subroutine
    call SY(IL, IU, BB, DD, AA, CC)
    
    ! Print results
    print *, 'Solution:'
    do I = 1, n
        print *, 'U_', I, ' = ', CC(I)
    end do

contains

subroutine SY(IL, IU, BB, DD, AA, CC)
    implicit none
    integer, intent(in) :: IL, IU
    real, dimension(IU), intent(inout) :: BB, DD, AA, CC
    integer :: I, J, LP
    real :: R
    
    ! Establish upper triangular matrix
    LP = IL + 1
    do I = LP, IU
        R = BB(I) / DD(I-1)
        DD(I) = DD(I) - R * AA(I-1)
        CC(I) = CC(I) - R * CC(I-1)
    end do
    
    ! Back substitution
    CC(IU) = CC(IU) / DD(IU)
    do I = LP, IU
        J = IU - I + IL
        CC(J) = (CC(J) - AA(J) * CC(J+1)) / DD(J)
    end do
end subroutine SY

end program tdma