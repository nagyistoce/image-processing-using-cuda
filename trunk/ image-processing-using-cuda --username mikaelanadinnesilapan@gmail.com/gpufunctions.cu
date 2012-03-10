#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cutil_inline.h>
#include "gpufunctions.h"


int iDivUp(int a, int b){
    return ((a % b) != 0) ? (a / b + 1) : (a / b);
}


__device__ unsigned int make_color(float r, float g, float b, float a){
    return
        ((int)(a * 255.0f) << 24) |
        ((int)(b * 255.0f) << 16) |
        ((int)(g * 255.0f) <<  8) |
        ((int)(r * 255.0f) <<  0);
}

__device__ float4 adjust_brightness (float4 rgba, float brightness){
	float4 a;
	a.x = rgba.x*(1.f - brightness)+brightness;
	a.y = rgba.y*(1.f - brightness)+brightness;
	a.z = rgba.z*(1.f - brightness)+brightness;
	a.w = rgba.w*(1.f - brightness)+brightness;
	return a;
}

__device__ float4 adjust_contrast (float4 rgba, float contrast){
	float4 a;
	a.x = pow(rgba.x, contrast);
	a.y = pow(rgba.y, contrast);
	a.z = pow(rgba.z, contrast);
	a.w = pow(rgba.w, contrast);
	return a;
}

/**************************************************
 Global variables for texture fetching and array
**************************************************/

texture<float, 2> texFLOAT;
texture<uchar4, 2, cudaReadModeNormalizedFloat> texImage;
texture<uchar4, 2, cudaReadModeNormalizedFloat> texUCHAR;

cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
	
cudaArray *a_Src, *b_Src, *d_tempArray, *d_array;

/*******************************************
 CUDA Init Functions for Texture and Array
*******************************************/

extern "C" 
cudaError_t CUDA_BindTexture ()
{
	cudaError_t error;

	error = cudaBindTextureToArray(texImage, d_array);
	return error;
}

extern "C" 
cudaError_t CUDA_UnbindTexture ()
{
	cudaError_t error;

	error = cudaUnbindTexture(texImage);

	return error;
}

extern "C" 
cudaError_t CUDA_MallocArray(uchar4 **h_Src, int imageW, int imageH)
{
    cudaError_t error;

    error = cudaMallocArray (&d_tempArray, &channelDesc, imageW, imageH );
    error = cudaMallocArray(&d_array, &channelDesc, imageW, imageH);
    error = cudaMemcpyToArray(d_array, 0, 0, *h_Src, imageW * imageH * sizeof(unsigned int), cudaMemcpyHostToDevice);	
    return error;
}

extern "C"
cudaError_t CUDA_FreeArray()
{
	cudaError_t error = cudaFreeArray(d_array);    
	error = cudaFreeArray(d_tempArray);    
    return error;
}


/*******************************************
   Image Processing Kernels
*******************************************/

#include "kernel_CopyImage.cu"
#include "kernel_Grayscale.cu"
#include "kernel_Brightness.cu"
#include "kernel_GammaCorrection.cu"
#include "kernel_MeanFilter.cu"
#include "kernel_SobelFilter.cu"
#include "kernel_Binarization.cu"
#include "kernel_HighPassFilter.cu"
#include "kernel_Invert.cu"
#include "kernel_BinaryErosion.cu"
#include "kernel_BinaryDilation.cu"
#include "kernel_GrayErosion.cu"
#include "kernel_GrayDilation.cu"
