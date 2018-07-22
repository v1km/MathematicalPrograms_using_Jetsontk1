//PROGRAM8
//THIS PROGRAM IS USED TO KNOW HOW MANY CUDA ENABLED DEVICES ARE
//THERE ON YOUR MACHINE AND LISTSING THEIR NAMES AND PROPERTIES.
//IT THEN FINDS A DEVICE WITH MAXIMUM MULTIPROCESSOR (bestdevice) AND SET IT
//FOR CUMPUTATION
#include<stdio.h>
#include<cuda.h>
int main()
{
	int i, DeviceCount, BestDevice, max=0;
	struct cudaDeviceProp properties;
	cudaError_t err;
	if(cudaSuccess!=(err=cudaGetDeviceCount(&DeviceCount)))
   {
   printf("\n Cuda Get device count Failed-error is-%s",(char*)cudaGetErrorString(err));
   DeviceCount=0;   // A machine without CUDA GPU can return 1 as an emulation device
   exit(EXIT_FAILURE);
   }
	printf("\nTotal %d CUDA devices found \n", DeviceCount);
	//List their names and properties
	for(i=0; i<DeviceCount; i++)
   {
           cudaGetDeviceProperties(&properties, i);
     printf("\ndevice= %d of name=%s ", i, properties.name );
	 printf("\nhas  multiprocessor=%d",properties.multiProcessorCount);
	 printf("\nhas  Max shared memory per block=%d",properties.sharedMemPerBlock);
	 printf("\nhas  Max block size =%d",properties.maxThreadsPerBlock);
	 getchar();
      if(properties.multiProcessorCount>max)
        {
            BestDevice=i;
           max=properties.multiProcessorCount;
		  printf("\n max=%d",max);
        }
   }
	//Get the properties of BestDevice now
	  cudaGetDeviceProperties(&properties, BestDevice);
	printf("\nBestDevice is=%s ", properties.name );
	//Set the Best device now
	  cudaSetDevice(BestDevice);
	//Here onwards all CUDA kernels will execute on BestDevice
	
  return 0;
}