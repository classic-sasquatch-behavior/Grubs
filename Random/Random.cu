#include"external_libs"





__global__ void seed_curand_xor( int size, int seed, curandState* states) {
	DIMS_1D(id);
	BOUNDS_1D(size);

	curand_init(seed, id, 0, &states[id]);
}




namespace on {
	
	On_Structure Random {

		curandState* Initialize::curand_xor(int size, int seed) {
			
			curandState* states;
			cudaMalloc((void**)&states, size * sizeof(curandState));
		
			sk::configure::kernel_1d(size);
			seed_curand_xor<<<LAUNCH>>>(size, seed, states);
			//SYNC_KERNEL(seed_curand_xor);

			return states;
		}

	}



}