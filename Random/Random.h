#pragma once

#include"external_libs.h"








namespace Substrate {

	namespace Random {
		
		namespace Initialize {
			
			static curandState* curand_xor(int size, int seed);
		};
	}
}