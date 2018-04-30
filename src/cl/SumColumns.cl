// sampler for image
__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;


// kernel sums over partial sums
__kernel void sumColumns(__read_only image2d_t in, __global float* out) 
{
	const int alphaIdx=get_global_id(0); // position for which calculation takes place
	const int w=get_image_width(in); // width of img

	float sum=0.0f;
	for(int x=0; x<w; x++)
	{
		float4 px=read_imagef(in, sampler, (int2)(x,alphaIdx));
		sum+=px.s0;
	}
	
	out[alphaIdx]=sum;
}

