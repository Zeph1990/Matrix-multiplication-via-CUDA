#include <stdio.h>

#include <stdlib.h>

#include <math.h>

#include <assert.h>

#include <time.h>

#include "cuda.h"

#include "cuda_runtime.h"

#include "device_launch_parameters.h"

#include "c_timer.h"



__global__ void matmatGPU(float *, float*, float*, int);
void randomMat(int, float ***);
void stampaMat(int , float ***);
void vettomat(float *, float ***, int);
void mattovet(float **, float *, int);

int main(){
	

	float **A, **B, **C,*A_v,*B_v,*C_v, *A_d, *B_d, *C_d;
	int size,N,NB,nrowb,ncolb;

	cudaEvent_t start, stop;
	float timeGPU;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);


	printf("Inserire la dimensione delle matrici: ");
	scanf("%d",&N);
	printf("\n Verranno allocate matrici di dimensioni %d*%d( %d elementi ciascuna ).\n",N,N,N*N);
	printf("Inserire il numero di thread per blocco desiderato:");
	scanf("%d",&NB);
	if (NB > 1024) {
		printf("Errore. Non è possibile disporre di più di 1024 thread per blocco.\n");
		system("PAUSE");
		return 0;
	}

	size = sizeof(float);
	
	cudaMalloc((void **)&A_d, size*N*N);
	cudaMalloc((void **)&B_d, size*N*N);
	cudaMalloc((void **)&C_d, size*N*N);
	randomMat(N, &A);
	randomMat(N, &B);
	int i = 0;
	C = (float**)calloc(N, sizeof(float*));
    for (i = 0; i<N; i++){
		C[i] = (float*)calloc(N, size);
	}


	A_v = (float *)calloc(N * N, size);
	B_v = (float *)calloc(N * N, size);
	C_v = (float *)calloc(N * N, size);

	mattovet(A,A_v,N);
	mattovet(B,B_v,N);

	cudaMemcpy(A_d, A_v, size*N*N, cudaMemcpyHostToDevice);
	cudaMemcpy(B_d, B_v, size*N*N, cudaMemcpyHostToDevice);

	int lgNB = log2((float)NB);
	if (lgNB % 2 == 0) {
		nrowb = sqrt((float)NB);
		ncolb = sqrt((float)NB);
	}
	else if (lgNB % 2 == 1){
		nrowb = pow(2.0, (int)(lgNB / 2) );
		ncolb = pow(2.0, (int)(lgNB / 2)+1 );
	}

	dim3 DimGrid(N/nrowb, N/ncolb);

	if ((N%DimGrid.x) != 0){ 
		DimGrid.x++;
	}

	if ((N%DimGrid.y) != 0){
		DimGrid.y++;
	}
	
	dim3 DimBlock(nrowb, ncolb);
	
	cudaEventRecord(start, 0);

	matmatGPU<<<DimGrid,DimBlock>>>(A_d, B_d, C_d, N);
	cudaThreadSynchronize();
	cudaEventRecord(stop, 0);

	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&timeGPU, start, stop);

	cudaEventDestroy(start);

	cudaEventDestroy(stop);

	
	printf("Tempo d'esecuzione per N=%d e con %d thread per blocco : %f secondi.\n",N,NB, timeGPU/1000);
	printf("Numero operazioni: %f GFLOPS\n",((2*pow((float)N,3)/(timeGPU/1000))/1000000000));
	cudaMemcpy(C_v, C_d, size*N*N, cudaMemcpyDeviceToHost);
	vettomat(C_v,&C,N);
	
	//stampaMat(N, &C);
	cudaFree(A_d);
	cudaFree(B_d); 
	cudaFree(C_d);

	free(A_v); 
	free(B_v); 
	free(C_v);
	
	free(A); 
	free(B); 
	free(C);

	system("PAUSE");
	return 0;

}
//CREA UNA MATRICE FLOAT RANDOM
void randomMat(int N, float ***A) {
	int i, j;

	*A = (float**)calloc(N, sizeof(float*));
	
	for (i = 0; i < N; i++){
		(*A)[i] = (float*)calloc(N, sizeof(float));

		for (j = 0; j < N; j++){

			((*A)[i][j]) =(100.0*rand() / RAND_MAX);
		}
	}
}

__global__ void matmatGPU(float *A, float *B, float *C, int N){

	int row, col, i;

	float Cloc = 0;

	row = blockIdx.y * blockDim.y + threadIdx.y;
	col = blockIdx.x * blockDim.x + threadIdx.x;

	if (row < N && col < N){

		for (i = 0; i<N; i++){
			Cloc = Cloc + A[row*N + i] * B[i*N + col];

		}

		C[row*N + col] = Cloc;

	}

}
void vettomat(float *V, float ***C,int N) {
	int i = 0;
	int j = 0;
	int x;
	for (x = 0; x<N*N; x++){
		(*C)[i][j] = V[x];
		j++;

		if (j >= N){
			i++;
			j = 0;
		}
	}

}
void mattovet(float **A, float *V,int N) {
	int x = 0;
	int i, j; 

	for (i = 0; i<N; i++){
		for (j = 0; j<N; j++){
			V[x] = A[i][j];
			x = x++;
		}
	}

}
//FUNZIONE PER LA STAMPA DELLA MATRICE
void stampaMat(int N, float ***A){
	int i, j;
for (i = 0; i<N; i++){
		for (j = 0; j<N; j++){
			printf("%lf   ", (*A)[i][j]);
		}
		printf("\n");
	}
}

