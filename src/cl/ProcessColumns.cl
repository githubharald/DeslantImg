// list of shearing angles
#define NUM_ALPHA_VALS 9
__constant float alphaVals[NUM_ALPHA_VALS] = { -1.0f, -0.75f, -0.5f, -0.25f, 0.0f, 0.25f, 0.5f, 0.75f, 1.0f };

// sampler for image
__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;


// kernel calculates partial sums for sheared images
__kernel void processColumns(__read_only image2d_t in, int maxShearedW, __global int* out) 
{
	const int h = get_image_height(in); // height of img
	const int x = get_global_id(0); // position for which calculation takes place	
	const int alphaIdx = get_global_id(1); // shear index
	const float alpha = alphaVals[alphaIdx]; // shear angle
	const int xOffset = x + alpha * (1-h); // offset into image
	
	// calculate length of the single continuous line of fg pixels, if it exists
	int edgeCtr = 0;
	int fgCtr = 0;
	int last = 0;
	for(int y = 0; y < h; y++)
	{
		const float4 px = read_imagef(in, sampler, (int2)(xOffset + alpha * y, y));
		const int val = px.s0 > 0.5f ? 1 : 0;
		
		edgeCtr = abs(last - val);
		fgCtr += val;
		
		last = val;
	}
	
	
	// write result
	out[maxShearedW * alphaIdx + x] = mad_sat(fgCtr, fgCtr, 0) * (edgeCtr <= 2 ? 1 : 0);
}

