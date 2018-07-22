//PROGRAM7
//THIS PROGRAM DEMONSTRATE USE OF PAGE LOCKED HOST MEMORY FOR SQUARING AN ARRAY USING A SIMPLE CUDA KERNEL
//Without measuring time
#include<stdio.h>
#include<cuda.h>

__global__ void SquareKernel(int *,int);

int main()
{
  int i; //loop variable
  int blockSize=128, blocks; //for cuda blocks
  cudaError_t err;//for error checking in cuda API
  int size=200;
  /**********************************/
   // Declare array pointer ha and allocate page locked host memory on CPU
      int *ha;// input array
	  if (cudaSuccess!=cudaHostAlloc(&ha, size*sizeof(int), cudaHostAllocPortable))
     {
         printf("\n problem in allocating page locked memory");
         exit(EXIT_FAILURE);
     }   
   
  /**************************************/
   //Declare array pointer ga intended for device/gpu
     int *ga;
     //Allocate space for array pointed by ga on GPU
         err=cudaMalloc((void **)&ga,size*sizeof(int));
	   //check memory allocation on GPU
	     if (cudaSuccess!=err)
	     {
		   printf("\n Memory allocation failed on GPU for ga");
		   printf("\n error is- %s", cudaGetErrorString(err));
		   exit(EXIT_FAILURE);
	     }
  /*******************************************************************/
   //Initialize ha
       for(i=0; i<size;i++)
	 {
	    ha[i]= (int) (rand()% 10);
	    //printf("\n%d", ha[i]);
	 }	  
  /*******************************************************************/
  //copy ha to ga from CPU to GPU
       if (cudaSuccess!=cudaMemcpy(ga,ha,size*sizeof(int),cudaMemcpyHostToDevice))
	{
		printf("\n Error in copying ha to ga");
		exit(EXIT_FAILURE);
	}
  /*********************************************************************/
 /***********************************************************************/
  //Compute number of cuda blocks needed  
	blocks=(int)(size/blockSize);
	if ((size%blockSize)>0)
		blocks++;
	printf("\n The number of blocks needed=%d", blocks);
 /**********************************************************************/
       //launch the cuda kernel 
        SquareKernel<<<blocks,blockSize>>>(ga, size); 
        cudaDeviceSynchronize();
  /*********************************************************************/
     //copy ga to ha from GPU to CPU
     if (cudaSuccess!=cudaMemcpy(ha,ga,size*sizeof(int),cudaMemcpyDeviceToHost))
	{
		printf("\n Error in copying ga to hb");
		exit(EXIT_FAILURE);
	}
 
 /**********************************************************************/
     // print values using hb
	 printf("\n values using GPU");
        for(i=0; i<size;i++)
          printf("\n%d", ha[i]);
  /**********************************************************************/ 
    //Do clean up
	   cudaFreeHost(ha);//Free page locked memory pointers
	   cudaFree(ga);// Device pointers
       getchar();
     return 0;
}


__global__ void SquareKernel(int *ga, int size)
{
	int i, z;	
	 i=(blockIdx.x*blockDim.x)+threadIdx.x; //for multi block	
	if(i<size)
	{
	  z=ga[i];
      ga[i]=z*z;
	    
	}
}


