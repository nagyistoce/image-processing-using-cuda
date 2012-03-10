#include <assert.h>
#include <time.h>
#include <vector_types.h>
#include <math.h>

unsigned int make_color_uint(int r, int g, int b, int a);
unsigned int make_color_float(float r, float g, float b, float a);
float4 adjust_brightness_float (float4 rgba, float brightness);
float4 adjust_contrast_float (float4 rgba, float contrast);

float brightnessCPU (float4 *src, unsigned int *dst, int imageW, int imageH, float brightness);
float gammaCorrectionCPU (float4 *src, unsigned int *dst, int imageW, int imageH, float gamma);
float copyImageCPU (float4 *src, unsigned int *dst, int imageW, int imageH, float brightness, float gamma, int adjust);
float grayscaleCPU (float4 *src, unsigned int *dst, int imageW, int imageH, float brightness, float gamma, int adjust);
float invertCPU (float4 *src, unsigned int *dst, int imageW, int imageH, float brightness, float gamma, int adjust);
float binarizeCPU (float4 *src, unsigned int *dst, int imageW, int imageH, int threshold, float brightness, float gamma, int adjust);
float smoothenCPU (float4 *src, unsigned int *dst, int imageW, int imageH, int radius, int iteration, float brightness, float contrast, int adjust);
float edgeDetectCPU (uchar4 *src, unsigned int *dst, int imageW, int imageH, float brightness, float gamma, int adjust);
float binaryErosionCPU (float4 *src, unsigned int *dst, int imageW, int imageH, int threshold, int iteration, float brightness, float gamma, int mask_w, int mask_h, int adjust);
float binaryDilationCPU (float4 *src, unsigned int *dst, int imageW, int imageH, int threshold, int iteration, float brightness, float gamma, int mask_w, int mask_h, int adjust);
float grayErosionCPU (float4 *src, unsigned int *dst, int imageW, int imageH, int threshold, int iteration, float brightness, float gamma, int mask_w, int mask_h, int adjust);
float grayDilationCPU (float4 *src, unsigned int *dst, int imageW, int imageH, int threshold, int iteration, float brightness, float gamma, int mask_w, int mask_h, int adjust);
float sharpenCPU (float4 *src, unsigned int *dst, int imageW, int imageH, float brightness, float contrast, int adjust);