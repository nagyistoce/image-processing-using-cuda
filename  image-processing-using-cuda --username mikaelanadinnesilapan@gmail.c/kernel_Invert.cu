__global__ void Invert (uint *dst, int imageW, int imageH, float brightness, float contrast)
{
	const int ix = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    const int iy = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;

	if(ix < imageW && iy < imageH){

		const float x = (float)ix + 0.5f;
		const float y = (float)iy + 0.5f;

		float4 fresult = tex2D(texImage, x, y);
		//adjust brightness
		float red = fresult.x * (1.f - brightness) + brightness;
		float green = fresult.y * (1.f - brightness) + brightness;
		float blue = fresult.z * (1.f - brightness) + brightness;

		//adjust contrast
		red = pow(red, contrast);
		green = pow(green, contrast);
		blue = pow(blue, contrast);
		
		red = 1.f - red;
		green = 1.f - green;
		blue = 1.f - blue;

		dst[imageW * iy + ix] =  make_color(red, green, blue, 1.f);
	}
}

extern "C" void invertWrapper (uint *dst, int imageW, int imageH, float brightness, float contrast)
{
	//for more effective kernel execution
	dim3 threads(BLOCKDIM_X, BLOCKDIM_Y);
	dim3 grid(iDivUp(imageW, BLOCKDIM_X), iDivUp(imageH, BLOCKDIM_Y));

	Invert<<<grid, threads>>>(dst, imageW, imageH, brightness, contrast);
	
}