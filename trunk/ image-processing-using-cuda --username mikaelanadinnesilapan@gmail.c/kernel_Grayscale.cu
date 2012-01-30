
__global__ void Grayscale ( uint *dst, int imageW, int imageH, float brightness, float contrast) 
{
    const int ix = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    const int iy = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;

	if(ix < imageW && iy < imageH){
		//Add half of a texel to always address exact texel centers
		const float x = (float)ix + 0.5f;
		const float y = (float)iy + 0.5f;
		float4 fresult = tex2D(texImage, x, y);
		float4 fnew = adjust_contrast(fresult, contrast);
		fnew = adjust_brightness(fnew, brightness);

		float gray = (fnew.x + fnew.y + fnew.z)/3;
        dst[imageW * iy + ix] = make_color(gray, gray, gray, 0);
	}
    
}

extern "C" void grayImageWrapper (uint *dst, int imageW, int imageH, float brightness, float contrast) 
{
	//for more effective kernel execution
	dim3 threads(BLOCKDIM_X, BLOCKDIM_Y);
	dim3 grid(iDivUp(imageW, BLOCKDIM_X), iDivUp(imageH, BLOCKDIM_Y));

	Grayscale<<<grid, threads>>>(dst, imageW, imageH, brightness, contrast);
}