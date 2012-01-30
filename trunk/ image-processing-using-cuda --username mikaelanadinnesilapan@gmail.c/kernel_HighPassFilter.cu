__device__ float convolve (float p00, float p01, float p02, 
					  float p10, float p11, float p12, 
					  float p20, float p21, float p22) 
{
	return (9*p11 -p00 -p01 -p02 -p10 -p12 -p20 -p21 -p22);    
}

__global__ void HighPassFilter(uint *dst, int imageW, int imageH, float brightness, float contrast)
{	
	
    const int ix = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    const int iy = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;

	if(ix < imageW && iy < imageH) {
		//Add half of a texel to always address exact texel centers
		const float x = (float)ix + 0.5f;
		const float y = (float)iy + 0.5f;
		float4 rgba;

		float pix00 = (tex2D( texImage, (float) x-1, (float) y-1 ).x);
		float pix01 = (tex2D( texImage, (float) x+0, (float) y-1 ).x);
		float pix02 = (tex2D( texImage, (float) x+1, (float) y-1 ).x);
		float pix10 = (tex2D( texImage, (float) x-1, (float) y+0 ).x);
		float pix11 = (tex2D( texImage, (float) x+0, (float) y+0 ).x);
		float pix12 = (tex2D( texImage, (float) x+1, (float) y+0 ).x);
		float pix20 = (tex2D( texImage, (float) x-1, (float) y+1 ).x);
		float pix21 = (tex2D( texImage, (float) x+0, (float) y+1 ).x);
		float pix22 = (tex2D( texImage, (float) x+1, (float) y+1 ).x);
			
		rgba.x = convolve(	pix00, pix01, pix02, 
								pix10, pix11, pix12,
								pix20, pix21, pix22 );
		 
		 pix00 = (tex2D( texImage, (float) x-1, (float) y-1 ).y);
		 pix01 = (tex2D( texImage, (float) x+0, (float) y-1 ).y);
		 pix02 = (tex2D( texImage, (float) x+1, (float) y-1 ).y);
		 pix10 = (tex2D( texImage, (float) x-1, (float) y+0 ).y);
		 pix11 = (tex2D( texImage, (float) x+0, (float) y+0 ).y);
		 pix12 = (tex2D( texImage, (float) x+1, (float) y+0 ).y);
		 pix20 = (tex2D( texImage, (float) x-1, (float) y+1 ).y);
		 pix21 = (tex2D( texImage, (float) x+0, (float) y+1 ).y);
		 pix22 = (tex2D( texImage, (float) x+1, (float) y+1 ).y);
			
		rgba.y = convolve(	pix00, pix01, pix02, 
								pix10, pix11, pix12,
								pix20, pix21, pix22 );

		 pix00 = (tex2D( texImage, (float) x-1, (float) y-1 ).z);
		 pix01 = (tex2D( texImage, (float) x+0, (float) y-1 ).z);
		 pix02 = (tex2D( texImage, (float) x+1, (float) y-1 ).z);
		 pix10 = (tex2D( texImage, (float) x-1, (float) y+0 ).z);
		 pix11 = (tex2D( texImage, (float) x+0, (float) y+0 ).z);
		 pix12 = (tex2D( texImage, (float) x+1, (float) y+0 ).z);
		 pix20 = (tex2D( texImage, (float) x-1, (float) y+1 ).z);
		 pix21 = (tex2D( texImage, (float) x+0, (float) y+1 ).z);
		 pix22 = (tex2D( texImage, (float) x+1, (float) y+1 ).z);
			
		rgba.z = convolve(	pix00, pix01, pix02, 
								pix10, pix11, pix12,
								pix20, pix21, pix22 );

		float4 fnew = adjust_contrast(rgba, contrast);
		fnew = adjust_brightness(fnew, brightness);

		dst[imageW * iy + ix] =	make_color(fnew.x, fnew.y, fnew.z, 1.f);
	
	}
}


extern "C" void highPassFilterWrapper (uint *dst, int imageW, int imageH, float brightness, float contrast)
{
	//for more effective kernel execution
	dim3 threads(BLOCKDIM_X, BLOCKDIM_Y);
	dim3 grid(iDivUp(imageW, BLOCKDIM_X), iDivUp(imageH, BLOCKDIM_Y));

	HighPassFilter<<<grid, threads>>>(dst, imageW, imageH, brightness, contrast);
}