
__device__ int getMedianIndex (int size)
{
	return (size % 2 == 0) ? (int)size/2 : (int)(size/2)+1;
}

__global__ void MedianFilter( RGBA *dst, int imageW, int imageH)
{

	//wag gawing shared!
	float3 window[BLOCKDIM_X* BLOCKDIM_Y][MEDIAN_WINDOW_SIZE];
	
    const int ix = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    const int iy = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
    int tid = threadIdx.x*blockDim.y+threadIdx.y;

    if(ix < imageW && iy < imageH){

		float4 fresult = {0,0,0,0};
		int index = 0;
		//Add half of a texel to always address exact texel centers
		const float x = (float)ix + 0.5f;
		const float y = (float)iy + 0.5f;
	
		
        for(float i = -MIN_MEDIAN_RADIUS; i <= MIN_MEDIAN_RADIUS; i++)
			for(float j = -MIN_MEDIAN_RADIUS; j <= MIN_MEDIAN_RADIUS; j++, index++)
			{	
				fresult = tex2D(texImage, x + i, y + j);
				window[tid][index].x = fresult.x;
				window[tid][index].y = fresult.y;
				window[tid][index].z = fresult.z;				
			}

		syncthreads();
		
		int i, j;
		float tmp = 0.f;

		for (i = 1; i < MEDIAN_WINDOW_SIZE; i++) 
		{
			j = i;
			while (j > 0 && window[tid][j - 1].x > window[tid][j].x) 
			{
				  tmp = window[tid][j].x;
				  window[tid][j].x = window[tid][j - 1].x;
				  window[tid][j - 1].x = tmp;
				  j--;
			}
			syncthreads();
		}

		for (i = 1; i < MEDIAN_WINDOW_SIZE; i++) 
		{
			j = i;
			while (j > 0 && window[tid][j - 1].y > window[tid][j].y) 
			{
				  tmp = window[tid][j].y;
				  window[tid][j].y = window[tid][j - 1].y;
				  window[tid][j - 1].y = tmp;
				  j--;
			}
			syncthreads();
		}

		for (i = 1; i < MEDIAN_WINDOW_SIZE; i++) 
		{
			j = i;
			while (j > 0 && window[tid][j - 1].z > window[tid][j].z) 
			{
				  tmp = window[tid][j].z;
				  window[tid][j].z = window[tid][j - 1].z;
				  window[tid][j - 1].z = tmp;
				  j--;
			}
			syncthreads();
		}

		int m = getMedianIndex(MEDIAN_WINDOW_SIZE);
		dst[imageW * iy + ix] = make_color(window[tid][m].x, window[tid][m].y, window[tid][m].z, 0);
	}
}

extern "C" void medianFilterWrapper (RGBA *dst, int imageW, int imageH)
{
	//for more effective kernel execution
	dim3 threads(BLOCKDIM_X, BLOCKDIM_Y);
	dim3 grid(iDivUp(imageW, BLOCKDIM_X), iDivUp(imageH, BLOCKDIM_Y));

	MedianFilter<<<grid, threads>>>(dst, imageW, imageH);
}