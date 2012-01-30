/*
**  Reduce noise by using Mean Filter with equal weights
*/

__global__ void MeanFilter ( uint *dst, int imageW, int imageH, int radius, float brightness, float contrast)
{
    const int ix = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    const int iy = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
    //Add half of a texel to always address exact texel centers
    const float x = (float)ix + 0.5f;
    const float y = (float)iy + 0.5f;

    if(ix < imageW && iy < imageH){

		float3 sum = {0,0,0};
		float4 fresult = {0,0,0,0};
		float count = 0.f;

        for( float i = -radius; i <= radius; i++)
            for( float j = -radius; j <= radius; j++)
            {
				fresult = tex2D(texImage, x + j, y + i);

				//adjust contrast
				float red = pow(fresult.x, contrast);
				float green = pow(fresult.y, contrast);
				float blue = pow(fresult.z, contrast);

				//adjust brightness
				red = red * (1.f - brightness) + brightness;
				green = green * (1.f - brightness) + brightness;
				blue = blue * (1.f - brightness) + brightness;

				sum.x += red;
				sum.y += green;
				sum.z += blue;
				count += 1.f;
            }

		sum.x /= count;
		sum.y /= count;
		sum.z /= count;

        //Write final result to global memory
        dst[imageW * iy + ix] = make_color(sum.x, sum.y, sum.z, 0);
    }
}

extern "C" void meanFilterWrapper (uint *dst, int imageW, int imageH, int radius, float brightness, float contrast) 
{
	//for more effective kernel execution
	dim3 threads(BLOCKDIM_X, BLOCKDIM_Y);
	dim3 grid(iDivUp(imageW, BLOCKDIM_X), iDivUp(imageH, BLOCKDIM_Y));

	MeanFilter<<<grid, threads>>>(dst, imageW, imageH, radius, brightness, contrast);
}