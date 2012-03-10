#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include <math.h>
#include <vector_types.h>

#define BLOCKDIM_X 8
#define BLOCKDIM_Y 8

#define MIN_MEAN_RADIUS 3 
#define MAX_MEAN_RADIUS(width) width/3

extern "C" void loadImageSDL ( uchar4 **dst, int *w, int *h, const char *file );
extern "C" void loadImageSDLFloat ( float4 **dst, int *w, int *h, const char *file );

//CUDA initialization functions

extern "C" cudaError_t CUDA_BindTexture ();
extern "C" cudaError_t CUDA_UnbindTexture ();
extern "C" cudaError_t CUDA_MallocArray(uchar4 **h_Src, int imageW, int imageH);
extern "C" cudaError_t CUDA_FreeArray();

//image processing CUDA kernels

extern "C" float copyImageWrapper (unsigned int *dst, int imageW, int imageH, float brightness, float contrast, int adjust); 
extern "C" float grayImageWrapper (unsigned int *dst, int imageW, int imageH, float brightness, float contrast, int adjust);

extern "C" float brightnessWrapper (unsigned int *dst, int imageW, int imageH, float brightness);
extern "C" float gammaCorrectionWrapper (unsigned int *dst, int imageW, int imageH, float gamma);

extern "C" float invertWrapper (unsigned int *dst, int imageW, int imageH, float brightness, float contrast, int adjust);
extern "C" float binarizationWrapper (unsigned int *dst, int imageW, int imageH, int threshold, float brightness, float contrast, int adjust);

extern "C" float meanFilterWrapper (unsigned int *dst, int imageW, int imageH, int radius, int iteration, float brightness, float contrast, int adjust);
extern "C" float sobelFilterWrapper (unsigned int *dst, int imageW, int imageH, float brightness, float contrast, int adjust);
extern "C" float highPassFilterWrapper (unsigned int *dst, int imageW, int imageH, int iteration, float brightness, float contrast, int adjust);

extern "C" float binaryErosionWrapper (unsigned int *dst, int imageW, int imageH, int threshold, int iteration, float brightness, float contrast, int mask_w, int mask_h, int adjust);
extern "C" float binaryDilationWrapper (unsigned int *dst, int imageW, int imageH, int threshold, int iteration, float brightness, float contrast, int mask_w, int mask_h, int adjust);
extern "C" float grayErosionWrapper (unsigned int *dst, int imageW, int imageH, int threshold, int iteration, float brightness, float contrast, int mask_w, int mask_h, int adjust);
extern "C" float grayDilationWrapper (unsigned int *dst, int imageW, int imageH, int threshold, int iteration, float brightness, float contrast, int mask_w, int mask_h, int adjust);

#endif