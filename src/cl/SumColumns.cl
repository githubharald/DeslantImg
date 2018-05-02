// kernel sums over partial sums
__kernel void sumColumns(__global int16* in, int maxShearedW, __global int* out) 
{
	const int alphaIdx = get_global_id(0);
	const int maxShearedW16 = maxShearedW / 16;
	
	int16 sum = (int16)(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
	for(int x = 0; x < maxShearedW16; x++)
	{	
		sum += in[maxShearedW16 * alphaIdx + x];
	}
	
	out[alphaIdx] = sum.s0 + sum.s1 + sum.s2 + sum.s3 + sum.s4 + sum.s5 + sum.s6 + sum.s7 + sum.s8 + sum.s9 + sum.sa + sum.sb + sum.sc + sum.sd + sum.se + sum.sf;
}

