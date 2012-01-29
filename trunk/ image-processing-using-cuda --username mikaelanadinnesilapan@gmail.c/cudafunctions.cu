#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "cudafunctions.h"


int iDivUp(int a, int b){
    return ((a % b) != 0) ? (a / b + 1) : (a / b);
}


__device__ uint make_color(float r, float g, float b, float a){
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
	return a;
}

__device__ float4 adjust_contrast (float4 rgba, float contrast){
	float4 a;
	a.x = pow(rgba.x, contrast);
	a.y = pow(rgba.y, contrast);
	a.z = pow(rgba.z, contrast);
	return a;
}

/**************************************************
 Global variables for texture fetching and array
**************************************************/

texture<uchar4, 2, cudaReadModeNormalizedFloat> texImage;
cudaChannelFormatDesc uchar4tex = cudaCreateChannelDesc<uchar4>();


cudaArray *a_Src;


/*******************************************
 CUDA Init Functions for Texture and Array
*******************************************/

extern "C"
cudaError_t CUDA_BindTextureToArray()
{
    return cudaBindTextureToArray(texImage, a_Src);
}

extern "C"
cudaError_t CUDA_UnbindTexture()
{
    return cudaUnbindTexture(texImage);
}

extern "C" 
cudaError_t CUDA_MallocArray(uchar4 **h_Src, int imageW, int imageH)
{
    cudaError_t error;

    error = cudaMallocArray(&a_Src, &uchar4tex, imageW, imageH);
    error = cudaMemcpyToArray(a_Src, 0, 0,
                              *h_Src, imageW * imageH * sizeof(uchar4),
                              cudaMemcpyHostToDevice
                              );
    return error;
}

extern "C"
cudaError_t CUDA_FreeArray()
{
	cudaError_t error = cudaFreeArray(a_Src);    
    return error;
}


/*******************************************
   Image Processing Kernels
*******************************************/

#include "kernel_CopyImage.cu"
#include "kernel_Grayscale.cu"
#include "kernel_MeanFilter.cu"
#include "kernel_MedianFilter.cu"
#include "kernel_SobelFilter.cu"
#include "kernel_Binarization.cu"
#include "kernel_HighPassFilter.cu"
#include "kernel_Gamma.cu"
#include "kernel_Brightness.cu"
#include "kernel_Invert.cu"
