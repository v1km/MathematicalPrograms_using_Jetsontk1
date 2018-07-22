//PROGRAM9
//THIS IS A SINGLE CUDA BLOCK REDUCTION EXAMPLE
//IT FIND OUT MINIMUM AMONG 200 VALUES USING A CUDA KERNEL
#include<stdio.h>
#include<cuda.h>
__global__ void ReductionKernel(int *,int* ,int);
__device__ void warpReduce(volatile int* , int );

int main()
{
	int * input,j; // input vector to be reduced
	int minimum; //for storing minimum value
	int size=200, blocksize=256, blocks;
	int sh;//shared memory required per block in bytes
	//Declare GPU pointer
	int *d_input,*d_output;// input pointer, output pointer	
	//compute shared memory needed in bytes/block it is bytes
	   //needed to store 256 integers
	   sh=blocksize*sizeof(int);
	//Allocate space for input vector on CPU
	 input=(int*)malloc(size*sizeof(int));
	 //Check memory allocations on CPU
     if((input==NULL))
     {
	    printf("\n Unable to allocate space on CPU for input array ");
	    exit(EXIT_FAILURE);
     }
	 /*******************************************************************/
   //Initialize input
       for(j=0; j<size;j++)
	 {
	    input[j]= (int) (rand()% 10);
	    printf("\n%d", input[j]);
	 }
	   getchar();
	   //input[3]=-5;
  /*******************************************************************/
	  //Allocate space for input array pointed by input on GPU        
	   //check memory allocation on GPU
	     if (cudaSuccess!=cudaMalloc((void **)&d_input,size*sizeof(int)))
	     {
			printf("\n Memory allocation failed on GPU for input array");			
			exit(EXIT_FAILURE);
	     }
		 if (cudaSuccess!=cudaMalloc((void **)&d_output,1*sizeof(int)))
	     {
			printf("\n Memory allocation failed on GPU for output array");			
			exit(EXIT_FAILURE);
	     }
	     //copy input array to GPU in d_in		 
        if (cudaSuccess!=cudaMemcpy(d_input,input,size*sizeof(int),cudaMemcpyHostToDevice))
	    {
		 printf("\n Error in copying input to d_input");
		 exit(EXIT_FAILURE);
	    }
	   //Compute number of blocks
	   blocks=1;
	     printf("\n The number of blocks needed=%d", blocks);
	   //Call to reduction kernel
	   ReductionKernel<<<blocks,blocksize,sh>>>(d_input,d_output,size);
	   cudaDeviceSynchronize();
	   //Copy minimum from d_output to minimum
	   if (cudaSuccess!=cudaMemcpy(&minimum,d_output,1*sizeof(int),cudaMemcpyDeviceToHost))
	   {
		printf("\n Error in copying d_output to minimum");
		exit(EXIT_FAILURE);
	   }
	   printf("\n minimum is =%d", minimum);
	   getchar();
	   //Do clean up
	     free(input);//CPU pointer
		 cudaFree(d_input);cudaFree(d_output);// GPU pointers
	   return 0;
}

__global__ void ReductionKernel(int *d_input, int* d_output,int size)
{
	int i=threadIdx.x;
	extern __shared__ int s[ ]; 
    //copy max in s[i]
      s[i]=INT_MAX; 
      __syncthreads();
      //copy the values from d_input in s[i] 
        if (i<size)
            s[i]=d_input[i];
          __syncthreads();
        //Do the reduction
		if (blockDim.x>=512)
	    {	
		if(i<256)
		{
			if(s[i]>s[i+256])
			{
				s[i]= s[i+256];			  
			}
		 }
		__syncthreads( );
	    }
        if (blockDim.x>=256)
	    {	
		if(i<128)
		{
			if(s[i]>s[i+128])
			{
				s[i]= s[i+128];			  
			}
		 }
		__syncthreads( );
	    }

	    if (blockDim.x>=128)
	    {
		if(i<64)
		{
			if(s[i]>s[i+64])
			{
				s[i]=s[i+64];					
			}
		}
		__syncthreads();
	    }
		
	  if(i<32)
	 {
		volatile int* s1=s;
		//This is the last warp thus unroll this part		
		if (blockDim.x>=64)		
		if(s1[i]>s1[i+32])
		{   //change the element
			s1[i]= s1[i+32];					
		}
		if (blockDim.x>=32)		
	    if(s1[i]>s1[i+16])
		{
			s1[i]= s1[i+16];
					
		}

		if (blockDim.x>=16)		
		if(s1[i]>s1[i+8])
		{
			s1[i]=s1[i+8];
				
		}
		if (blockDim.x>=8)		
		if(s1[i]>s1[i+4])
		{
			s1[i]=s1[i+4];
				
		}
		if (blockDim.x>=4)		
		if(s1[i]>s1[i+2])
		{
			s1[i]= s1[i+2];					
			
		}
		if (blockDim.x>=2)		
		if(s1[i]>s1[i+1])
		{
			s1[i]= s1[i+1];			
			
		}		
	}	
	  //single block reduction is over
	  //write minimum at zeroth location of d_output using thread with id=0
       if (i== 0) 
       d_output[blockIdx.x] = s[0];
}

