__device__ float Sobel (float p00, float p01, float p02, 
					  float p10, float p11, float p12, 
					  float p20, float p21, float p22) 
{
	float Gx = p02 + 2*p12 + p22 - p00 - 2*p10 - p20;
    float Gy = p00 + 2*p01 + p02 - p20 - 2*p21 - p22;
    float G = (abs(Gx)+abs(Gy));
    if ( G < 0 ) return 0.f; else if ( G > 1.f ) return 1.f;
    return G;

}

__global__ void CopyGrayscale( uint *dst, int imageW, int imageH ) 
{
    const int ix = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    const int iy = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;

	if(ix < imageW && iy < imageH){
		//Add half of a texel to always address exact texel centers
		const float x = (float)ix + 0.5f;
		const float y = (float)iy + 0.5f;
		float4 fresult = tex2D(texImage, x, y);
		float gray = (fresult.x + fresult.y + fresult.z)/3;		
        dst[imageW * iy + ix] = make_color(gray, gray, gray, 0);
	}    
}


__global__ void SobelFilter(uint *dst, int imageW, int imageH)
{	
	
    const int ix = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    const int iy = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
    //Add half of a texel to always address exact texel centers
    const float x = (float)ix + 0.5f;
    const float y = (float)iy + 0.5f;

	float pix00 = (tex2D( texImage, (float) x-1, (float) y-1 ).x);
    float pix01 = (tex2D( texImage, (float) x+0, (float) y-1 ).x);
    float pix02 = (tex2D( texImage, (float) x+1, (float) y-1 ).x);
    float pix10 = (tex2D( texImage, (float) x-1, (float) y+0 ).x);
    float pix11 = (tex2D( texImage, (float) x+0, (float) y+0 ).x);
    float pix12 = (tex2D( texImage, (float) x+1, (float) y+0 ).x);
    float pix20 = (tex2D( texImage, (float) x-1, (float) y+1 ).x);
    float pix21 = (tex2D( texImage, (float) x+0, (float) y+1 ).x);
    float pix22 = (tex2D( texImage, (float) x+1, (float) y+1 ).x);
		
	float sobel = Sobel(	pix00, pix01, pix02, 
							pix10, pix11, pix12,
							pix20, pix21, pix22 );

	dst[imageW * iy + ix] =	make_color(sobel, sobel, sobel, 1.f);
	
}


extern "C" void sobelFilterWrapper (uint *dst, int imageW, int imageH)
{
	//for more effective kernel execution
	dim3 threads(BLOCKDIM_X, BLOCKDIM_Y);
	dim3 grid(iDivUp(imageW, BLOCKDIM_X), iDivUp(imageH, BLOCKDIM_Y));

	CopyGrayscale<<<grid, threads>>>(dst, imageW, imageH);
	SobelFilter<<<grid, threads>>>(dst, imageW, imageH);
}