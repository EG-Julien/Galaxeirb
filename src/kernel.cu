#include "cuda.h"

__global__ void kernel_saxpy( int n, float * Masse, float * PosX, float * PosY, float * PosZ, float * VelX, float * VelY, float * VelZ, float * accX, float * accY, float * accZ ) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if ( i < n ) { 
		float aX = 0;
		float aY = 0;
		float aZ = 0;
		int j;
	

	
		for(j=0; j<n; j++){	
	
			if(!(i==j)){
			
				float deltaX = (PosX[j]-PosX[i]); 
				float deltaY = (PosY[j]-PosY[i]); 
				float deltaZ = (PosZ[j]-PosZ[i]);

				float Dij = sqrtf((deltaX*deltaX) + (deltaY*deltaY) + (deltaZ*deltaZ));

				if ( Dij < 1.0 ) Dij = 1.0;

				float coef = 10 * 1 * (1/(Dij*Dij*Dij)) * Masse[j];
				 
				aX += deltaX * coef;
				aY += deltaY * coef; 
				aZ += deltaZ * coef;
				

			}
	
		 

		}

	
		accX[i] = aX; 
		accY[i] = aY; 
		accZ[i] = aZ; 
	}
}

void saxpy( int nblocks, int nthreads, int n, float * deviceSrc1, float * deviceSrc2, float * deviceSrc3, float * deviceSrc4, float  * deviceSrc5, float  * deviceSrc6, float  * deviceSrc7, float * deviceSrc8, float * deviceSrc9, float * deviceSrc10 ) {
	kernel_saxpy<<<nblocks, nthreads>>>( n, deviceSrc1, deviceSrc2, deviceSrc3, deviceSrc4, deviceSrc5, deviceSrc6, deviceSrc7, deviceSrc8, deviceSrc9, deviceSrc10 );
}
