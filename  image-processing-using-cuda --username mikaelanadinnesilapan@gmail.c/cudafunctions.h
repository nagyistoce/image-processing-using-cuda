#ifndef CUDA_FUNCTIONS_H
#define CUDA_FUNCTIONS_H

#include <math.h>

typedef unsigned int uint;

#define MIN_MEAN_RADIUS 3 
#define MAX_MEAN_RADIUS(width) width/3

#define MIN_MEDIAN_RADIUS 3
#define MEDIAN_WINDOW_SIZE (int)(MIN_MEDIAN_RADIUS*MIN_MEDIAN_RADIUS) 

#define BLOCKDIM_X 8
#define BLOCKDIM_Y 8

extern "C" void loadImageSDL ( uchar4 **dst, int *w, int *h, char *file );

//CUDA initialization functions

extern "C" cudaError_t CUDA_BindTextureToArray();
extern "C" cudaError_t CUDA_UnbindTexture();
extern "C" cudaError_t CUDA_MallocArray(uchar4 **h_Src, int imageW, int imageH);
extern "C" cudaError_t CUDA_FreeArray();

//image processing kernels

extern "C" void copyImageWrapper (uint *dst, int imageW, int imageH, float brightness, float contrast); 
extern "C" void grayImageWrapper (uint *dst, int imageW, int imageH, float brightness, float contrast);
extern "C" void invertWrapper (uint *dst, int imageW, int imageH, float brightness, float contrast);
extern "C" void binarizationWrapper (uint *dst, int imageW, int imageH, int threshold, float brightness, float contrast);
extern "C" void meanFilterWrapper (uint *dst, int imageW, int imageH, int radius, float brightness, float contrast);
extern "C" void sobelFilterWrapper (uint *dst, int imageW, int imageH, float brightness, float contrast);
extern "C" void highPassFilterWrapper (uint *dst, int imageW, int imageH, float brightness, float contrast);


#endif