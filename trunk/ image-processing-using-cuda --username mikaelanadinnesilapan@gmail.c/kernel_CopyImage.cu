#include <shrUtils.h>

__global__ void Copy ( uint *dst, int imageW, int imageH, float brightness, float contrast) 
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

		dst[imageW * iy + ix] =  make_color(fnew.x, fnew.y, fnew.z, 1.f);        
    }
}

extern "C" double copyImageWrapper (uint *dst, int imageW, int imageH, float brightness, float contrast) 
{
	//for more effective kernel execution
	dim3 threads(BLOCKDIM_X, BLOCKDIM_Y);
	dim3 grid(iDivUp(imageW, BLOCKDIM_X), iDivUp(imageH, BLOCKDIM_Y));

	double kernel_time = 0.0;
	shrDeltaT(0);
	Copy<<<grid, threads>>>(dst, imageW, imageH, brightness, contrast);
	kernel_time = shrDeltaT(0);

	return kernel_time;

}