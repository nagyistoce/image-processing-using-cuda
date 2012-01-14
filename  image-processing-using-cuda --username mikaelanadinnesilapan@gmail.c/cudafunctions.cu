#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "cudafunctions.h"

////////////////////////////////////////////////////////////////////////////////
// Helper functions
////////////////////////////////////////////////////////////////////////////////
float Max(float x, float y){
    return (x > y) ? x : y;
}

float Min(float x, float y){
    return (x < y) ? x : y;
}

int iDivUp(int a, int b){
    return ((a % b) != 0) ? (a / b + 1) : (a / b);
}


__device__ float lerpf(float a, float b, float c){
    return a + (b - a) * c;
}

__device__ float vecLen(float4 a, float4 b){
    return (
        (b.x - a.x) * (b.x - a.x) +
        (b.y - a.y) * (b.y - a.y) +
        (b.z - a.z) * (b.z - a.z)
    );
}

__device__ RGBA make_color(float r, float g, float b, float a){
    return
        ((int)(a * 255.0f) << 24) |
        ((int)(b * 255.0f) << 16) |
        ((int)(g * 255.0f) <<  8) |
        ((int)(r * 255.0f) <<  0);
}

// convert floating point rgba color to 32-bit integer
__device__ unsigned int rgbaFloatToInt(float4 rgba)
{
    rgba.x = __saturatef(rgba.x);   // clamp to [0.0, 1.0]
    rgba.y = __saturatef(rgba.y);
    rgba.z = __saturatef(rgba.z);
    rgba.w = __saturatef(rgba.w);
    return (unsigned int(rgba.w * 255.0f) << 24) | (unsigned int(rgba.z * 255.0f) << 16) | (unsigned int(rgba.y * 255.0f) << 8) | unsigned int(rgba.x * 255.0f);
}



////////////////////////////////////////////////////////////////////////////////
// Global data handlers and parameters
////////////////////////////////////////////////////////////////////////////////
//Texture reference and channel descriptor for image texture
texture<uchar4, 2, cudaReadModeNormalizedFloat> texImage;
cudaChannelFormatDesc uchar4tex = cudaCreateChannelDesc<uchar4>();

//CUDA array descriptor
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
