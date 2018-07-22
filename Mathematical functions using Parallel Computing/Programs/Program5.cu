//PROGRAM4
//THIS IS PROGRAM 4- VECTOR MATRIX MULTIPLICATION WITH MEASURING TIME
//USING CUDA EVENTS AND USES SHARED MEMORY

#include<stdio.h>
#include<cuda.h>

__global__ void VMMulti(float *, float*, float*, int, int);
void VMMultiSerial(float *, float* , float *, int , int );

int main()
{
  int i; //loop variable
  int blockSize=128, blocks; //for cuda blocks

  cudaError_t err;//for error checking in cuda API
  /*************************************************************/
   //Declare and create cuda events and a float for 
   float ElapsedTimeS, ElapsedTimeP;
   cudaEvent_t start, stop; //Declare events
   cudaEventCreate(&start);  //Create events
   cudaEventCreate(&stop); 
  /************************************************************/
 // pointers to vector V , matrix M and their result R=V*M on Host/CPU
     float *V, *M, *R; 
 // pointers to vector V, matrix M and for R, on Device/GPU gV and gM
  float * gV, *gM, *gR;
/***************************************************************/
 //Define their sizes
  int Vsize=3; //Vector size (1 x Vsize)
  int Mcols=5000; //Columns in M (Vsize x Mcols)
 /*************************************************************/
  //Declare a variable for computing amount of shared memory needed per block
  int Sh;
 /***************************************************************/
  //Allocate space on Host for V, M, R on CPU
    V=(float*)malloc(Vsize*sizeof(float));
    M=(float*)malloc(Vsize*Mcols*sizeof(float));
	R=(float*)malloc(Mcols*sizeof(float));
  //check the allocations
  if( (V==NULL)||(M==NULL)||(R==NULL))
  {
	  printf("\n Unable to allocate space on CPU for either V/ M/ R ");
	  exit(EXIT_FAILURE);
  }
 /****************************************************************/  
  //Allocate space on Device for gV, gM, gR and check error if any
      err=cudaMalloc((void**)&gV,Vsize*sizeof(float));
	   //check the allocation
	     if (cudaSuccess!=err)
		 {
			 printf("\n Memory allocation failed on GPU for gV");
			 printf("\n error is- %s", cudaGetErrorString(err));
			 exit(EXIT_FAILURE);
		 }
	 err=cudaMalloc((void **)&gM,Vsize*Mcols*sizeof(float));
	   //check the allocation
	     if (cudaSuccess!=err)
		 {
			 printf("\n Memory allocation failed on GPU for gM");
			 printf("\n error is- %s", cudaGetErrorString(err));
			 exit(EXIT_FAILURE);
		 }
	 err=cudaMalloc((void **)&gR,Mcols*sizeof(float));
	   //check the allocation
	     if (cudaSuccess!=err)
		 {
			 printf("\n Memory allocation failed on GPU for gR");
			 printf("\n error is- %s", cudaGetErrorString(err));
			 exit(EXIT_FAILURE);
		 }
  /***********************************************************************/

  //Initialize V and M with random values  
     for(i=0; i<Vsize;i++)
	 {
	    V[i]= (float) (rand()% 10);
		//printf("\n%f",V[i]);
	 }
	 //M is assumed to be stored in column major
     for(i=0; i<(Vsize*Mcols);i++)
	 {
		 M[i]= (float) (rand()% 10);
	    //printf("\n%f", M[i]);
	 }
  /***********************************************************************/
  //Copy V and M  from CPU to GPU
	if (cudaSuccess!=cudaMemcpy(gV,V,Vsize*sizeof(float),cudaMemcpyHostToDevice))
	{
		printf("\n Error in copying V to gV");
		exit(EXIT_FAILURE);
	}
	if (cudaSuccess!=cudaMemcpy(gM,M,Vsize*Mcols*sizeof(float),cudaMemcpyHostToDevice))
    {
		printf("\n Error in copying M to gM");
		exit(EXIT_FAILURE);
	}
 /***********************************************************************/
  //Compute number of cuda blocks needed to compute Mcols
  //elements of R=V*M
	blocks=(int)(Mcols/blockSize);
	if ((Mcols%blockSize)>0)
		blocks++;
	printf("\n The number of blocks needed=%d", blocks);
 /**********************************************************************/
	//Compute number of bytes needed to store gV in Shared memory
	Sh=Vsize*sizeof(float);
 /*********************************************************************/
  //Call the cuda kernel VMMUlti for computing R=V*M on GPU
  // and measure the time using events start and stop
	cudaEventRecord(start,0);
	    VMMulti<<<blocks,blockSize, Sh>>>(gV, gM,gR,Vsize, Mcols);
	    cudaThreadSynchronize(); //synchronize CPU and GPU
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	//Compute the elapsed time
	cudaEventElapsedTime(&ElapsedTimeP,start, stop);
	printf("\n Parallel time is=%f milliseconds", ElapsedTimeP);
/***********************************************************************/
  //call the serial function and measure the time using events start and stop
	cudaEventRecord(start,0);
	      VMMultiSerial(V, M, R, Vsize, Mcols);
    cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	//Compute the elapsed time
	cudaEventElapsedTime(&ElapsedTimeS,start, stop);
	printf("\n Serial Execution time is=%f milliseconds", ElapsedTimeS);
	printf("\n Speed up is=%f ", ElapsedTimeS/ElapsedTimeP);

/************************************************************************/
  //Copy Result back to CPU in R
    if (cudaSuccess!=cudaMemcpy(R,gR,Mcols*sizeof(float),cudaMemcpyDeviceToHost))
    {
		printf("\n Error in copying gR to R");
		exit(EXIT_FAILURE);
	}
	/*********************************************************************/
  //Print result on CPU
  	/* for(i=0; i<Mcols;i++)
	 {
	    printf("\n%f",R[i]);
	 }*/
 /***********************************************************************/
 //Do  clean up
	free(V);free(M);free(R); //Host pointers
	cudaFree(gV); cudaFree(gM); cudaFree(gR); //Device pointers
	cudaEventDestroy(start);cudaEventDestroy(stop); //cuda events
 /***********************************************************************/
  getchar();
   return 0;
}


__global__ void VMMulti(float *gV, float*gM, float *gR, int Vsize, int Mcols)
{
	int i, j;
	float sum=0.0;
	i=(blockIdx.x*blockDim.x)+threadIdx.x;  //for multi block
   //Allocate shared memory
	  extern __shared__ float sV[ ];
   //Copy gV to SV from global to shared memory
	  if (i<Vsize)
	     sV[i]=gV[i];
	  __syncthreads();
	//gM is stored in column major order
	if(i<Mcols)
	{
	  for(j=0;j<Vsize;j++)
		  sum=sum+(sV[j]*gM[j+(i*Vsize)]);
	     __syncthreads();
	    gR[i]=sum;
	}
}


//Serial C function to compute R=V*M
void VMMultiSerial(float *V, float* M, float *R, int Vsize, int Mcols)
{
	int i, j;
	float sum;
	//compute Mcols dot products
	for(i=0;i<Mcols;i++)
	{
		sum=0.0;
	 for(j=0;j<Vsize;j++)
	 {
		 sum=sum+(V[j]*M[j+(i*Vsize)]);

	 }
	    R[i]=sum;
	}
}