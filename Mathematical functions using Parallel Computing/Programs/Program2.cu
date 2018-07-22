//PROGRAM01
//THIS PROGRAM DEMONSTRATE SQUARING AN ARRAY USING A SIMPLE CUDA KERNEL
//WITH MEASURING CPU AND GPU TIME AND SPEED UP
#include<stdio.h>
#include<cuda.h>
#include<time.h>

__global__ void SquareKernel(int *,int);
void SquareSerial(int*, int );

int main()
{
  int i; //loop variable
  int blockSize=128, blocks; //for cuda blocks
  cudaError_t err;//for error checking in cuda API
  int size=200;
  /***********************************/
  float timespentCPU, timespentGPU;
  clock_t start1, stop1;
  /**********************************/
    cudaEvent_t start, stop; 
	cudaEventCreate(&start); //Creates an event object 
     cudaEventCreate(&stop);
  /*************************************/
   // Declare array pointer ha and hb on Host/CPU
      int *ha;// input array
      int *hb;// output array
   //Allocate space for array pointed by ha on CPU
     ha=(int*)malloc(size*sizeof(int));
     hb=(int*)malloc(size*sizeof(int));
     //Check memory allocations on CPU
     if((ha==NULL)||(hb==NULL))
     {
	printf("\n Unable to allocate space on CPU for ha/hb ");
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
	cudaEventRecord(start, 0); //Timestamp, zero –default stream
        SquareKernel<<<blocks,blockSize>>>(ga, size); 
        cudaDeviceSynchronize();
		cudaEventRecord(stop, 0); //Timestamp
        cudaEventSynchronize(stop); 
		cudaEventElapsedTime(&timespentGPU, start, stop); 
		printf("\n timespent on GPU=%f",timespentGPU);
  /*********************************************************************/
     //copy ga to ha from GPU to CPU
     if (cudaSuccess!=cudaMemcpy(hb,ga,size*sizeof(int),cudaMemcpyDeviceToHost))
	{
		printf("\n Error in copying ga to hb");
		exit(EXIT_FAILURE);
	} 
 /**********************************************************************/
     // print values using hb
	 printf("\n values using GPU");
        for(i=0; i<size;i++)
          printf("\n%d", hb[i]);
  /**********************************************************************/
   //call the serial function SquareSerial
		start1=clock();
         SquareSerial(&ha[0],size);
		 stop1=clock();
		 timespentCPU = ((float)(stop1 - start1))/CLOCKS_PER_SEC;
		 printf("\n timespent on CPU=%f",timespentCPU);
		 getchar();
  /*********************************************************************/
    /**********************************************************************/
     // print values using ha
		 printf("\n values using CPU");
        for(i=0; i<size;i++)
          printf("\n%d", ha[i]);
  /**********************************************************************/
		printf("\n speed up=%f",(float)(timespentCPU/timespentGPU));

  /**********************************************************************/
    //Do clean up
	   free(ha);free(hb); //host pointers
	   cudaFree(ga); // Device pointers
	   //Destroy events
	    cudaEventDestroy(start); 
        cudaEventDestroy(stop); 
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


//Serial C function to square
void SquareSerial(int* ha, int size)
{
	int i, j;
	
	for(i=0;i<size;i++)
	{
        j=ha[i];
		ha[i]=j*j;
	 
	}
}