#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "cblas.h"

void initial(double *A, double *B, int N);
double error(double *x, double *y, int N);
void matrix_multiplication(double *A, double *B, double *C, int N);
void dgemm_row(double *A, double *B, double *C, int N);
//void dgemm_col(double *A, double *B, double *C, int N);

int main()
{
	printf("\n");
	int N;
	printf(" Input size N = ");
	scanf("%d", &N);
	printf(" N = %d \n\n", N);

	double *A, *B, *C1, *C2, *exact;
	double t1, t2, T1, T2;

	A = (double *) malloc(N*N*sizeof(double));
	B = (double *) malloc(N*N*sizeof(double));
	C1 = (double *) malloc(N*N*sizeof(double));
	C2 = (double *) malloc(N*N*sizeof(double));
	exact = (double *) malloc(N*N*sizeof(double));

	initial(A, B, N);

	t1 = clock();
	matrix_multiplication(A, B, exact, N);
	t2 = clock();
	T1 = 1.0*(t2 - t1)/CLOCKS_PER_SEC;

	t1 = clock();
	dgemm_row(A, B, C1, N);
	t2 = clock();
	T2 = 1.0*(t2 - t1)/CLOCKS_PER_SEC;

//	dgemm_col(A, B, C2, N);

	printf(" Error of dgemm_row = %e \n", error(C1, exact, N*N));
	printf(" Origin matrix multiplication dgemm times : %f \n", T1);
	printf(" Blas level-3 matrix multiplication dgemm times : %f \n", T2);
//	printf(" Error of dgemm_col = %e \n", error(C2, exact, N*N));

	printf("\n");
	return 0;
}

void initial(double *A, double *B, int N)
{
	int i;

	// initail A
	for (i=0; i<N*N; i++)	A[i] = 1.0*i;
	// initial B
	for (i=0; i<N*N; i++)	B[i] = 1.0*(i+1);
}

double error(double *x, double *y, int N)
{
	int i;
	double e, temp;
	e = 0.0;
	for (i=0; i<N; i++)
	{
		temp = fabs(x[i] - y[i]);
		if (temp > e)	e = temp;
	}

	return e;
}

// C = alpha * op(A) * op(B) + beta * op(C)
void matrix_multiplication(double *A, double *B, double *C, int N)
{
	int i, j, k;
	double alpha, beta, temp;

	alpha = 1.0;
	beta = 0.0;

	for (i=0; i<N; i++)
	{
		for (j=0; j<N; j++)
		{
			temp = 0.0;
			for (k=0; k<N; k++)
			{
				temp += A[N*i+k] * B[N*k+j];
			}
			C[N*i+j] = temp + beta * C[N*i+j];
		}
	}
}

// C = alpha * op(A) * op(B) + beta * op(C)
void dgemm_row(double *A, double *B, double *C, int N)
{
	double alpha, beta;
	alpha = 1.0;
	beta = 0.0;
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N, alpha, A, N, B, N, beta, C, N);
}

/*
// C = alpha * op(A) * op(B) + beta * op(C)
void dgemm_col(double *A, double *B, double *C, int N)
{
	double alpha, beta;
	alpha = 1.0;
	beta = 0.0;
	cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, N, N, N, alpha, B, N, A, N, beta, C, N);
}
*/

