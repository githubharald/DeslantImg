// list of shearing angles
#define NUM_ALPHA_VALS 9
__constant float alphaVals[NUM_ALPHA_VALS] = { -1.0f, -0.75f, -0.5f, -0.25f, 0.0f, 0.25f, 0.5f, 0.75f, 1.0f };

// sampler for image
__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;


// kernel calculates partial sums for sheared images
__kernel void processColumns(__read_only image2d_t in, __global int* out) 
{
	const int h = get_image_height(in); // height of img
	const int alphaIdx = get_global_id(1); // shear index
	const float alpha = alphaVals[alphaIdx]; // shear angle
	const int xOffset = get_global_id(0) + alpha * (1-h); // offset into image
	
	// calculate length of the single continuous line of fg pixels, if it exists
	int edgeCtr = 0;
	int fgCtr = 0;
	int last = 0;
	for(int y = 0; y < h; y++)
	{
		const int val = read_imagei(in, sampler, (int2)(xOffset + alpha * y, y)).s0 != 0;
		edgeCtr = abs(last - val);
		fgCtr += val;
		
		last = val;
	}
	
	// init group sum
	__local int localSum;
	if(get_local_id(0) == 0)
	{
		localSum = 0;
	}
	
	// wait until local memory is initialized
	barrier(CLK_LOCAL_MEM_FENCE);
	
	// add result of work item to result of work group
	atomic_add(&localSum, mad_sat(fgCtr, fgCtr, 0) * ((int)(edgeCtr <= 2)));
	
	// wait until local memory has final result
	barrier(CLK_LOCAL_MEM_FENCE);
	
	// write result
	if(get_local_id(0) == 0)
	{
		out[get_num_groups(0) * alphaIdx + get_group_id(0)] = localSum;
	}
}

