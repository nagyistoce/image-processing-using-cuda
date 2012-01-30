__global__ void GammaCorrection (uint *dst, int imageW, int imageH, float brightness, float gamma)
{
	const int ix = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    const int iy = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;

	if(ix < imageW && iy < imageH){

		const float x = (float)ix + 0.5f;
		const float y = (float)iy + 0.5f;

		float4 fresult = tex2D(texImage, x, y);
		float red = fresult.x * (1.f - brightness) + brightness;
		float green = fresult.y * (1.f - brightness) + brightness;
		float blue = fresult.z * (1.f - brightness) + brightness;

		red = pow(red, gamma);
		green = pow(green, gamma);
		blue = pow(blue, gamma);

		dst[imageW * iy + ix] =  make_color(red, green, blue, 1.f);
	}
}


// if gamma is 0..1 , the dark intensities are stretched up
// if gamma is 1..5 , the high intensities are stretched down

extern "C" void gammaCorrectionWrapper (uint *dst, int imageW, int imageH, float brightness, float gamma)
{
	dim3 threads(BLOCKDIM_X, BLOCKDIM_Y);
	dim3 grid(iDivUp(imageW, BLOCKDIM_X), iDivUp(imageH, BLOCKDIM_Y));

	GammaCorrection<<<grid, threads>>>(dst, imageW, imageH, brightness, gamma);
}