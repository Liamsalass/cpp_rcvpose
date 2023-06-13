
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "JACOBI.h"
#include "lmshorn.h"

/**
***	Calculates optimal (least mean square) transformation
***	from P1 to P2, i.e. P2[i] = A*P1[i].
***	Assumes that the lists P1 and P2 are ordered,
***	i.e. P2[i] corresponds to P1[i].
**/	

void lmshorn(double (*P1)[3], double (*P2)[3], int n, double A[4][4])
{
	int i, j, nrot, max_eigind ;
	double C1[3]={0.0,0.0,0.0}, C2[3]={0.0,0.0,0.0},
		Sxx=0.0, Sxy=0.0, Sxz=0.0, 
		Syx=0.0, Syy=0.0, Syz=0.0, 
		Szx=0.0, Szy=0.0, Szz=0.0, 
		N[5][5], D[5], V[5][5], max_eigval,
		q0, q1, q2, q3, R[3][3], T[3] ;
	

	/*	calculate centroids	*/
	for (i=0; i<n; i++)
	{
		for (j=0; j<3; j++)
		{
			C1[j] += P1[i][j] ;
			C2[j] += P2[i][j] ;
		}
	}

	for (j=0; j<3; j++)
	{
		C1[j] /= n ;
		C2[j] /= n ;
	}
	
	/*	translate point sets to cetroids	*/	
	for (i=0; i<n; i++)
	{
		for (j=0; j<3; j++)
		{
			P1[i][j] -= C1[j] ;
			P2[i][j] -= C2[j] ;
		}
	}

	/*	calculate elements of M matrix	*/
	for (i=0; i<n; i++)
	{
		Sxx += P1[i][0]*P2[i][0] ;
		Sxy += P1[i][0]*P2[i][1] ;
		Sxz += P1[i][0]*P2[i][2] ;

		Syx += P1[i][1]*P2[i][0] ;
		Syy += P1[i][1]*P2[i][1] ;
		Syz += P1[i][1]*P2[i][2] ;
	
		Szx += P1[i][2]*P2[i][0] ;
		Szy += P1[i][2]*P2[i][1] ;
		Szz += P1[i][2]*P2[i][2] ;
	}

	/*	generate N matrix 
	**
	**	Note : matrix indeces start at 1, to be NR compatible
	*/
	N[1][1]=Sxx+Syy+Szz; N[1][2]=Syz-Szy; N[1][3]=Szx-Sxz; N[1][4]=Sxy-Syx;
	N[2][1]=Syz-Szy; N[2][2]=Sxx-Syy-Szz; N[2][3]=Sxy+Syx; N[2][4]=Szx+Sxz;
	N[3][1]=Szx-Sxz; N[3][2]=Sxy+Syx; N[3][3]=-Sxx+Syy-Szz; N[3][4]=Syz+Szy;
	N[4][1]=Sxy-Syx; N[4][2]=Szx+Sxz; N[4][3]=Syz+Szy; N[4][4]=-Sxx-Syy+Szz; 
	/*	find eigenvectors	*/
	myjacobi(N,4,D,V,&nrot) ;	

	/*	find max eigenvalue	*/
	max_eigind = 1 ;
	max_eigval = D[max_eigind] ;
	for (i=2; i<=4; i++)
	{
		if (D[i] > max_eigval)
		{
			max_eigind = i ;
			max_eigval = D[max_eigind] ;
		}
	}

	/*	generate optimal rotation matrix
	**
	**	Note : start matrix indexing at 0 again
	*/
	q0 = V[1][max_eigind] ;
	q1 = V[2][max_eigind] ;
	q2 = V[3][max_eigind] ;
	q3 = V[4][max_eigind] ;

	R[0][0] = q0*q0 + q1*q1 - q2*q2 - q3*q3 ;
	R[0][1] = 2*(q1*q2 - q0*q3) ;
	R[0][2] = 2*(q1*q3 + q0*q2) ;
	R[1][0] = 2*(q1*q2 + q0*q3) ;
	R[1][1] = q0*q0 + q2*q2 - q1*q1 -q3*q3 ;
	R[1][2] = 2*(q2*q3 - q0*q1) ;
	R[2][0] = 2*(q1*q3 - q0*q2) ;
	R[2][1] = 2*(q2*q3 + q0*q1) ;
	R[2][2] = q0*q0 + q3*q3 - q1*q1 -q2*q2 ;
 
	/*	calculate translation	*/
	T[0] = C2[0] - (R[0][0]*C1[0] + R[0][1]*C1[1] + R[0][2]*C1[2]) ;
	T[1] = C2[1] - (R[1][0]*C1[0] + R[1][1]*C1[1] + R[1][2]*C1[2]) ;
	T[2] = C2[2] - (R[2][0]*C1[0] + R[2][1]*C1[1] + R[2][2]*C1[2]) ;

	/*	compose into matrix	*/
	for (i=0; i<3; i++)
	{
		for (j=0; j<3; j++)
		{
			A[i][j] = R[i][j] ;
		}
	}
	
	for (j=0; j<3; j++)
	{
		A[j][3] = T[j] ;
		A[3][j] = 0.0 ;
	}

	A[3][3] = 1.0 ;

	/*	restore original point sets */	
	for (i=0; i<n; i++)
	{
		for (j=0; j<3; j++)
		{
			P1[i][j] += C1[j] ;
			P2[i][j] += C2[j] ;
		}
	}
}

void lmshorn2(
        double (*P1)[3],        /*      point set 1     */
        double (*P2)[3],        /*      point set 2	*/
        int n,              	/*      size of point sets  */
        double C1[3],           /*      centroid of point P1    */
        double C2[3],           /*      centroid of P2          */
        double Sxx, double Sxy, double Sxz,     /* partial sums : assumes */
        double Syx, double Syy, double Syz,     /* only P2 has been     */
        double Szx, double Szy, double Szz,     /* translated to centroid */
        double A[4][4])        /*      result          */

{
	int i, j, nrot, max_eigind ;
	double N[5][5], D[5], V[5][5], max_eigval,
		q0, q1, q2, q3, R[3][3], T[3] ;
	
	/*	adjust elements of M matrix to reflect the fact that 	*/
	/*	only P1 was translated to centroid :			*/
	/*								*/
	/*		sum( (A-k) * B ) = sum (A*B) - k*sum(B)		*/
	/*								*/
	for (i=0; i<n; i++)
	{
		P2[i][0] -= C2[0] ;
		P2[i][1] -= C2[1] ;
		P2[i][2] -= C2[2] ;

		Sxx -= C1[0]*P2[i][0] ;
		Sxy -= C1[0]*P2[i][1] ;
		Sxz -= C1[0]*P2[i][2] ;

		Syx -= C1[1]*P2[i][0] ;
		Syy -= C1[1]*P2[i][1] ;
		Syz -= C1[1]*P2[i][2] ;
	
		Szx -= C1[2]*P2[i][0] ;
		Szy -= C1[2]*P2[i][1] ;
		Szz -= C1[2]*P2[i][2] ;
	}

	/*	generate N matrix 
	**
	**	Note : matrix indeces start at 1, to be NR compatible
	*/
	N[1][1]=Sxx+Syy+Szz; N[1][2]=Syz-Szy; N[1][3]=Szx-Sxz; N[1][4]=Sxy-Syx;
	N[2][1]=Syz-Szy; N[2][2]=Sxx-Syy-Szz; N[2][3]=Sxy+Syx; N[2][4]=Szx+Sxz;
	N[3][1]=Szx-Sxz; N[3][2]=Sxy+Syx; N[3][3]=-Sxx+Syy-Szz; N[3][4]=Syz+Szy;
	N[4][1]=Sxy-Syx; N[4][2]=Szx+Sxz; N[4][3]=Syz+Szy; N[4][4]=-Sxx-Syy+Szz; 
	/*	find eigenvectors	*/
	myjacobi(N,4,D,V,&nrot) ;	

	/*	find max eigenvalue	*/
	max_eigind = 1 ;
	max_eigval = D[max_eigind] ;
	for (i=2; i<=4; i++)
	{
		if (D[i] > max_eigval)
		{
			max_eigind = i ;
			max_eigval = D[max_eigind] ;
		}
	}

	/*	generate optimal rotation matrix
	**
	**	Note : start matrix indexing at 0 again
	*/
	q0 = V[1][max_eigind] ;
	q1 = V[2][max_eigind] ;
	q2 = V[3][max_eigind] ;
	q3 = V[4][max_eigind] ;

	R[0][0] = q0*q0 + q1*q1 - q2*q2 - q3*q3 ;
	R[0][1] = 2*(q1*q2 - q0*q3) ;
	R[0][2] = 2*(q1*q3 + q0*q2) ;
	R[1][0] = 2*(q1*q2 + q0*q3) ;
	R[1][1] = q0*q0 + q2*q2 - q1*q1 -q3*q3 ;
	R[1][2] = 2*(q2*q3 - q0*q1) ;
	R[2][0] = 2*(q1*q3 - q0*q2) ;
	R[2][1] = 2*(q2*q3 + q0*q1) ;
	R[2][2] = q0*q0 + q3*q3 - q1*q1 -q2*q2 ;
 
	/*	calculate translation	*/
	T[0] = C2[0] - (R[0][0]*C1[0] + R[0][1]*C1[1] + R[0][2]*C1[2]) ;
	T[1] = C2[1] - (R[1][0]*C1[0] + R[1][1]*C1[1] + R[1][2]*C1[2]) ;
	T[2] = C2[2] - (R[2][0]*C1[0] + R[2][1]*C1[1] + R[2][2]*C1[2]) ;

	/*	compose into matrix	*/
	for (i=0; i<3; i++)
	{
		for (j=0; j<3; j++)
		{
			A[i][j] = R[i][j] ;
		}
	}
	
	for (j=0; j<3; j++)
	{
		A[j][3] = T[j] ;
		A[3][j] = 0.0 ;
	}

	A[3][3] = 1.0 ;

	/*	restore original P2 */	
	for (i=0; i<n; i++)
	{
		for (j=0; j<3; j++)
		{
			P2[i][j] += C2[j] ;
		}
	}
}
