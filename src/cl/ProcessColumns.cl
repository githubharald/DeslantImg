// list of shearing angles
#define NUM_ALPHA_VALS 9
__constant float alphaVals[NUM_ALPHA_VALS] = { -1.0f, -0.75f, -0.5f, -0.25f, 0.0f, 0.25f, 0.5f, 0.75f, 1.0f };

// sampler for image
__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;


// kernel calculates partial sums for sheared images
__kernel void processColumns(__read_only image2d_t in, __write_only image2d_t out) 
{
	const int x=get_global_id(0); // position for which calculation takes place
	const int h=get_image_height(in); // height of img
	const int alphaIdx=get_global_id(1); // shear index
	const float alpha=alphaVals[alphaIdx]; // shear angle
	const int xOffset=alpha*(1-h);
	
	// calculate first and last fg pixel and number of fg pixels in sheared image
	bool firstSet=false;
	int firstFG=0;
	int lastFG=0;
	int sumFG=0;
	for(int y=0; y<h; y++)
	{
		// read pixel
		const float4 px=read_imagef(in, sampler, (int2)(xOffset+x+alpha*y,y));
		
		// fg pixel (white)
		if(px.s0>0.5f)
		{
			sumFG++; // one more fg pixel
			lastFG=y; // move last fg pixel one down
			if(!firstSet) // set first fg pixel if not yet set
			{
				firstFG=y;
				firstSet=true;
			}
		}
	}
	
	// write result
	float res=0.0f;
	const int dist=lastFG-firstFG+1;
	if(sumFG==dist)
	{
		res=sumFG*sumFG;
	}
	write_imagef(out, (int2)(x,alphaIdx), (float4)(res,0,0,0));
}

