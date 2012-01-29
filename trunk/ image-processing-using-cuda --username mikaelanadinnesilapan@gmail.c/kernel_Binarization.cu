
__global__ void Binarize(uint *dst, int imageW, int imageH, int threshold)
{	
	
    const int ix = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    const int iy = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
    
	int pix = dst[imageW * iy + ix] & 0xff;
	if(pix > threshold) dst[imageW * iy + ix] =	make_color(1.f, 1.f, 1.f, 1.f);
	else dst[imageW * iy + ix] = make_color(0.f, 0.f, 0.f, 1.f);
	
}

__global__ void Thresholding (uint *dst, int imageW, int imageH, int *threshold)
{

	/*********************************************************************************
	Since the dst is already grayscale, choose the threshold using the iterative method
	Algo:
	1. choose initial threshold, I chose 100.
	2. Since there will be background and object pixels, get the sum of each group.
	ex: if pix (> T), background += pix;
	3. Compute for the average of both sets.
	4. The new threshold is the average of the two sets. 
	5. Repeat 2-4 until the new threshold matches the previous threshold. :)
	**********************************************************************************/

	int init_T = 100;
	int new_T = 200;
	int bg_Sum = 0;
	int obj_Sum = 0;
	int bg_Av = 0;
	int obj_Av = 0;
	int pix = 0;
	int bg_Ctr = 0;
	int obj_Ctr = 0;

	while(init_T != new_T)
	{
		for(int i=0; i<imageW; i++)
		{
			for(int j=0; j<imageH; j++)
			{
				pix = (dst[j*imageW+i]) & 0xff;
				if(pix > init_T) {
					bg_Sum += pix;
					++bg_Ctr;
				}
				else {
					obj_Sum += pix;
					++obj_Ctr;
				}
			}
		}

		bg_Av = bg_Sum/bg_Ctr;
		obj_Av = obj_Sum/obj_Ctr;

		new_T = (bg_Av + obj_Av)/2; 

	}
	
	*threshold = new_T;
}


extern "C" void binarizationWrapper (uint *dst, int imageW, int imageH, int threshold, float brightness, float contrast)
{
	//for more effective kernel execution
	dim3 threads(BLOCKDIM_X, BLOCKDIM_Y);
	dim3 grid(iDivUp(imageW, BLOCKDIM_X), iDivUp(imageH, BLOCKDIM_Y));

	Grayscale<<<grid, threads>>>(dst, imageW, imageH, brightness, contrast);
	Binarize<<<grid, threads>>>(dst, imageW, imageH, threshold);
	
}