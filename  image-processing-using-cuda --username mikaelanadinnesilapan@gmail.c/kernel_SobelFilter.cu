__global__ void GrayscaleUchar( Pixel *dst, int imageW, int imageH ) 
{
    const int ix = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    const int iy = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;

	if(ix < imageW && iy < imageH){
		//Add half of a texel to always address exact texel centers
		const float x = (float)ix + 0.5f;
		const float y = (float)iy + 0.5f;
		float4 fresult = tex2D(texImage, x, y);
		float gray = (fresult.x + fresult.y + fresult.z)/3;
		Pixel grayPix = ((Pixel)gray)*255;
        dst[imageW * iy + ix] = grayPix;
	}    
}


__global__ void SobelFilter(Pixel *src, RGBA *dst, int imageW, int imageH)
{	
	/*
    const int ix = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    const int iy = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
    //Add half of a texel to always address exact texel centers
    const float x = (float)ix + 0.5f;
    const float y = (float)iy + 0.5f;

	float4 pixel = {0,0,0,0};
	float window [BLOCKDIM_X*BLOCKDIM_Y][9];
	int id = threadIdx.y*blockDim.y+threadIdx.x;
	int index = 0;

	for(int i=x-1; i<=x+1; i++)
	{
		for(int j=y-1; j<=y+1; j++)
		{
			pixel = tex2D(texImage, i, j);
			window[id][index++] = pixel.x; 
		}
	}

	
	dst[imageW * iy + ix] = Sobel(window[id][0], window[id][3], window[id][6], 
										window[id][1], window[id][4], window[id][7], 
										window[id][2], window[id][5], window[id][8]);
	*/
}


extern "C" void sobelFilterWrapper (Pixel *gray, RGBA *dst, int imageW, int imageH)
{

	/*cudaMallocArray(&gray_Src, &uchartex, imageW, imageH);
    cudaMemcpyToArray(gray_Src, 0, 0,
                              *h_Src, imageW * imageH * sizeof(uchar4),
                              cudaMemcpyHostToDevice
                              );
	*/

	//for more effective kernel execution
	dim3 threads(BLOCKDIM_X, BLOCKDIM_Y);
	dim3 grid(iDivUp(imageW, BLOCKDIM_X), iDivUp(imageH, BLOCKDIM_Y));

	SobelFilter<<<grid, threads>>>(gray, dst, imageW, imageH);
}