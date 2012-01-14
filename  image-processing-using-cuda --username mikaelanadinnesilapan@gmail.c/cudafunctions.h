#ifndef CUDA_FUNCTIONS_H
#define CUDA_FUNCTIONS_H

#include <math.h>

typedef unsigned int RGBA;
typedef unsigned char Pixel;

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

extern "C" void copyImageWrapper (RGBA *dst, int imageW, int imageH);
extern "C" void grayImageWrapper (RGBA *dst, int imageW, int imageH); 
extern "C" void meanFilterWrapper (RGBA *dst, int imageW, int imageH, int radius);
extern "C" void medianFilterWrapper (RGBA *dst, int imageW, int imageH);
extern "C" void sobelFilterWrapper (Pixel *gray, RGBA *dst, int imageW, int imageH);

#endif