#define NUM_ALPHA_VALS 9
__constant float alphaVals[NUM_ALPHA_VALS] = { -1.0f, -0.75f, -0.5f, -0.25f, 0.0f, 0.25f, 0.5f, 0.75f, 1.0f };

// kernel sums over partial sums
__kernel void sumColumns(__global int4* in, __global float* out) 
{
	// calculate result for work item
	const int id = get_global_id(0);
	const int4 tmp4 = in[id];
	const int2 tmp2 = tmp4.s01 + tmp4.s23;
	const int res = tmp2.s0 + tmp2.s1;
	
	// write result to local memory
	__local int alphaRes[NUM_ALPHA_VALS];
	alphaRes[id] = res;

	// return best result
	barrier(CLK_LOCAL_MEM_FENCE);
	if(id == 0)
	{
		int bestIdx = 0;
		int bestVal = 0;
		for(int i = 0; i < NUM_ALPHA_VALS; ++i)
		{
			const int currVal = alphaRes[i];
			if(currVal > bestVal)
			{
				bestVal = currVal;
				bestIdx = i;
			}
		}
		*out = -alphaVals[bestIdx]; // return negative value because operation performed is inverse to shear transform
	}
}

