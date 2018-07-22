//PROGRAM5
//THIS PROGRAM DEMONSTRATE VECTOR-MATRIX MULTIPLICATION USING GPU CONSTANT MEMORY
//IT STORES VECTOR V IN GPU CONSTANT MEMORY
#include<stdio.h>
#include<cuda.h>

__global__ void VMMulti(float*, float*, int, int);
void VMMultiSerial(float *, float* , float *, int , int );
__constant__ float gV[3];
int main()
{
  int i; //loop variable
  int blockSize=128, blocks; //for cuda blocks
  cudaError_t err;//for error checking in cuda API
 // pointers to vector V , matrix M and their result R=V*M on Host/CPU
     float *V, *M, *R; 
 // pointers to vector V, matrix M and for R, on Device/GPU gV and gM
  float *gM, *gR;
/***************************************************************/
 //Define their sizes
  int Vsize=3; //Vector size (1 x Vsize)
  int Mcols=5000; //Columns in M (Vsize x Mcols)

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
  //Allocate space on Device for gM, gR and check error if any
     
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
  //Copy V in GPU CONSTANT MEMORY
	
	 err=cudaMemcpyToSymbol(gV,V,Vsize*sizeof(float));
	if(err!=cudaSuccess)
	{
		printf("\n Error in copying V in constant GPU memory ");
		printf("\n Error is %s",cudaGetErrorString(err) );
		exit(EXIT_FAILURE);
    }

	//Copy  M  from CPU to GPU
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
  //Call the cuda kernel VMMUlti for computing R=V*M on GPU
	VMMulti<<<blocks,blockSize>>>( gM,gR,Vsize, Mcols);
	cudaDeviceSynchronize(); //synchronize CPU and GPU
/***********************************************************************/
  //call the serial function
	//VMMultiSerial(V, M, R, Vsize, Mcols);
/************************************************************************/
  //Copy Result back to CPU in R
    if (cudaSuccess!=cudaMemcpy(R,gR,Mcols*sizeof(float),cudaMemcpyDeviceToHost))
    {
		printf("\n Error in copying gR to R");
		exit(EXIT_FAILURE);
	}
	/*********************************************************************/
   //Print result on CPU
  	 for(i=0; i<Mcols;i++)
	 {
	    printf("\n%f",R[i]);
	 }
	/**********************************************************************/
	 //Do  clean up
	free(V);free(M);free(R); //Host pointers
	cudaFree(gM); cudaFree(gR); //Device pointers	
   /***********************************************************************/ 
  getchar();
   return 0;
}


__global__ void VMMulti( float*gM, float *gR, int Vsize, int Mcols)
{
	int i, j;
	float sum=0.0;
	i=(blockIdx.x*blockDim.x)+threadIdx.x;  
	//gM is stored in column major order
	if(i<Mcols)
	{
	  for(j=0;j<Vsize;j++)
	  {
		  sum=sum+(gV[j]*gM[j+(i*Vsize)]);//Note gV is stored in GPU constant memory
		 
	  }
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