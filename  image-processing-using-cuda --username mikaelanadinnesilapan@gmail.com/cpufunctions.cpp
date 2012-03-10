#include "cpufunctions.h"
#include <stdio.h>
#include <stdlib.h>

unsigned int make_color_uint(int r, int g, int b, int a){
    return (
        ((int)(a) << 24) |
        ((int)(b) << 16) |
        ((int)(g) <<  8) |
        ((int)(r) <<  0) );
}

unsigned int make_color_float(float r, float g, float b, float a){
    return
        ((int)(a * 255.0f) << 24) |
        ((int)(b * 255.0f) << 16) |
        ((int)(g * 255.0f) <<  8) |
        ((int)(r * 255.0f) <<  0);
}


float4 adjust_brightness_float (float4 rgba, float brightness){
	float4 a;
	a.x = rgba.x*(1.f - brightness)+brightness;
	a.y = rgba.y*(1.f - brightness)+brightness;
	a.z = rgba.z*(1.f - brightness)+brightness;
	a.w = rgba.w*(1.f - brightness)+brightness;
	return a;
}

float4 adjust_contrast_float (float4 rgba, float contrast){
	float4 a;
	a.x = pow(rgba.x, contrast);
	a.y = pow(rgba.y, contrast);
	a.z = pow(rgba.z, contrast);
	a.w = pow(rgba.w, contrast);
	return a;
}

unsigned char Sobel (unsigned char p00, unsigned char p01, unsigned char p02, 
					  unsigned char p10, unsigned char p11, unsigned char p12, 
					  unsigned char p20, unsigned char p21, unsigned char p22) 
{
	int Gx = p02 + 2*p12 + p22 - p00 - 2*p10 - p20;
    int Gy = p00 + 2*p01 + p02 - p20 - 2*p21 - p22;
    int G = (int)(abs(Gx)+abs(Gy));
    if ( G < 0 ) return 0; else if ( G > 255 ) return 255;
    return G;

}

float Highpass (float p00, float p01, float p02, 
					  float p10, float p11, float p12, 
					  float p20, float p21, float p22) 
{
	float result = (9*p11 -p00 -p01 -p02 -p10 -p12 -p20 -p21 -p22);
	if(result< 0.f) return 0.f; else if(result>1.f) return 1.f;
	return result;
}

float brightnessCPU (float4 *src, unsigned int *dst, int imageW, int imageH, float brightness) 
{
	int index=0;
	float4 fnew = {0, 0, 0, 0};

	clock_t start, stop;
	float t = 0.0;

	assert((start = clock())!=-1);
	
	for(int i=0; i<imageW; i++)
	{
		for(int j=0; j<imageH; j++)
		{
			index = (imageW)*j+i;					

			fnew.x = src[index].x*(1.f - brightness)+brightness;
			fnew.y = src[index].y*(1.f - brightness)+brightness;
			fnew.z = src[index].z*(1.f - brightness)+brightness;

			dst[index] = make_color_float(
							fnew.x,
							fnew.y, 
							fnew.z, 
							1.0f
						);			
		}
	}
	

	stop = clock();
	t = (float) (stop-start)/CLOCKS_PER_SEC;

	return t;
}

float gammaCorrectionCPU (float4 *src, unsigned int *dst, int imageW, int imageH, float gamma) 
{
	int index=0;
	float4 fnew = {0, 0, 0, 0};

	clock_t start, stop;
	float t = 0.0;

	assert((start = clock())!=-1);
	
	for(int i=0; i<imageW; i++)
	{
		for(int j=0; j<imageH; j++)
		{
			index = (imageW)*j+i;					

			fnew.x = pow(src[index].x, gamma);
			fnew.y = pow(src[index].y, gamma);
			fnew.z = pow(src[index].z, gamma);

			dst[index] = make_color_float(
							fnew.x,
							fnew.y, 
							fnew.z, 
							1.0f
						);			
		}
	}
	

	stop = clock();
	t = (float) (stop-start)/CLOCKS_PER_SEC;

	return t;
}

float copyImageCPU (float4 *src, unsigned int *dst, int imageW, int imageH, float brightness, float contrast, int adjust)
{
	int index=0;
	float4 fnew = {0, 0, 0, 0};

	clock_t start, stop;
	float t = 0.0;

	assert((start = clock())!=-1);

	if(adjust){
		for(int i=0; i<imageW; i++)
		{
			for(int j=0; j<imageH; j++)
			{
				index = (imageW)*j+i;			
								
				fnew = adjust_contrast_float(src[index], contrast);
				fnew = adjust_brightness_float(fnew, brightness);

				dst[index] = make_color_float(
								fnew.x,
								fnew.y, 
								fnew.z, 
								fnew.w
							);			
			}
		}
	}else{
		for(int i=0; i<imageW; i++)
		{
			for(int j=0; j<imageH; j++)
			{
				index = (imageW)*j+i;					
				dst[index] = make_color_float(
								src[index].x,
								src[index].y, 
								src[index].z, 
								src[index].w
							);			
			}
		}
	}

	stop = clock();
	t = (float) (stop-start)/CLOCKS_PER_SEC;

	return t;
}

float grayscaleCPU (float4 *src, unsigned int *dst, int imageW, int imageH, float brightness, float contrast, int adjust)
{
	int index=0;
	float gray=0;
	float4 fnew = {0, 0, 0, 0};
	
	clock_t start, stop;
	float t = 0.0;

	/* Start timer */
	assert((start = clock())!=-1);

	if(adjust){
		for(int i=0; i<imageW; i++)
		{
			for(int j=0; j<imageH; j++)
			{
				index = (imageW)*j+i;

				fnew = adjust_contrast_float(src[index], contrast);
				fnew = adjust_brightness_float(fnew, brightness);

				gray = (fnew.x + fnew.y + fnew.z)/3;
				dst[index] = make_color_float(gray, gray, gray, 1.0f);			

			}
		}
	}else{
		for(int i=0; i<imageW; i++)
		{
			for(int j=0; j<imageH; j++)
			{
				index = (imageW)*j+i;
				gray = (src[index].x + src[index].y + src[index].z)/3;
				dst[index] = make_color_float(gray, gray, gray, 1.0f);			

			}
		}
	}
	/* Stop timer */
	stop = clock();
	t = (float) (stop-start)/CLOCKS_PER_SEC;

	return t;
}


float invertCPU (float4 *src, unsigned int *dst, int imageW, int imageH, float brightness, float contrast, int adjust)
{
	int index=0;	
	float4 fnew = {0, 0, 0, 0};

	clock_t start, stop;
	float t = 0.0;

	/* Start timer */
	assert((start = clock())!=-1);

	if(adjust){
		for(int i=0; i<imageW; i++)
		{
			for(int j=0; j<imageH; j++)
			{
				index = (imageW)*j+i; 

				fnew = adjust_contrast_float(src[index], contrast);
				fnew = adjust_brightness_float(fnew, brightness);

				dst[index] = make_color_float(1.0-fnew.x, 1.0-fnew.y, 1.0-fnew.z, 1.0f);
			}
		}
	}else{
		for(int i=0; i<imageW; i++)
		{
			for(int j=0; j<imageH; j++)
			{
				index = (imageW)*j+i; 
				dst[index] = make_color_float(1.0-src[index].x, 1.0-src[index].y, 1.0-src[index].z, 1.0f);
			}
		}
	}
	/* Stop timer */
	stop = clock();
	t = (float) (stop-start)/CLOCKS_PER_SEC;

	return t;
}


float binarizeCPU (float4 *src, unsigned int *dst, int imageW, int imageH, int threshold, float brightness, float contrast, int adjust)
{
	int index=0;
	int pixel;
	
	clock_t start, stop;
	float t = 0.0;

	/* Start timer */
	assert((start = clock())!=-1);

	grayscaleCPU(src, dst, imageW, imageH, brightness, contrast, adjust);

	for(int i=0; i<imageW; i++)
	{
		for(int j=0; j<imageH; j++)
		{
			index = (imageW)*j+i; 
			pixel = ((dst)[index]) & 0xff;
			
			if(pixel > threshold){
				dst[index] = make_color_float(1.f, 1.f, 1.f , 1.f);
			}else{
				dst[index] = make_color_float(0.f, 0.f, 0.f, 0.f);
			}
		}
	}

	/* Stop timer */
	stop = clock();
	t = (float) (stop-start)/CLOCKS_PER_SEC;

	return t;
}

float smoothenCPU (float4 *src, unsigned int *dst, int imageW, int imageH, int radius, int iteration, float brightness, float contrast, int adjust)
{
	float pixel_x = 0;
	float pixel_y = 0;
	float pixel_z = 0;

	clock_t start, stop;
	float t = 0.0;
	float count = 1.f;
	int index = 0;
	float4 *temp = (float4*)malloc(sizeof(float4)*imageW*imageH);

	/* Start timer */
	assert((start = clock())!=-1);

	if(adjust){
		for(int i=0; i<imageW; i++)
		{
			for(int j=0; j<imageH; j++)
			{
				index = (imageW)*j+i; 
				temp[index] = adjust_contrast_float(src[index], contrast);
				temp[index] = adjust_brightness_float(temp[index], brightness);
			}
		}

		count = 1.f;
		pixel_x = pixel_y = pixel_z = 0;

		for(int i=0; i<imageW; i++)
		{
			for(int j=0; j<imageH; j++)
			{
				for (int m = i - radius ; m <= i + radius; m++){			
					for (int n = j - radius ; n <= j + radius; n++){		
						if(m > imageW || m <  0 || n > imageH || n < 0) {
							//lampas sa bound ng image
						}else{						
							pixel_x += temp[(imageW)*n+m].x;
							pixel_y += temp[(imageW)*n+m].y;
							pixel_z += temp[(imageW)*n+m].z;
							printf("%.2f %.2f %.2f\n", pixel_x, pixel_y, pixel_z);
							count += 1.f;
						}
					}
				}

				pixel_x /= count;
				pixel_y /= count;
				pixel_z /= count;

				dst[(imageW)*j+i] = make_color_float(pixel_x, pixel_y, pixel_z, 1.f);				

			}
		}

	}else{
		for(int i=0; i<imageW; i++)
		{
			for(int j=0; j<imageH; j++)
			{
				
				count = 1.f;
				pixel_x = pixel_y = pixel_z = 0;

				for (int m = i - radius ; m <= i + radius; m++){			
					for (int n = j - radius ; n <= j + radius; n++){		
						if(m > imageW || m <  0 || n > imageH || n < 0) {
							//lampas sa bound ng image
						}else{						
							pixel_x += src[(imageW)*n+m].x;
							pixel_y += src[(imageW)*n+m].y;
							pixel_z += src[(imageW)*n+m].z;
							count += 1.f;
						}
					}
				}

				pixel_x /= count;
				pixel_y /= count;
				pixel_z /= count;

				dst[(imageW)*j+i] = make_color_float(pixel_x, pixel_y, pixel_z, 1.f);

			}
		}

	}
	
	/* Stop timer */
	stop = clock();
	t = (float) (stop-start)/CLOCKS_PER_SEC;

	return t;
}

float edgeDetectCPU (uchar4 *src, unsigned int *dst, int imageW, int imageH, float brightness, float contrast, int adjust)
{
	int index=0;
	int pixel = 0, gray = 0;
	uchar4 *gray_src = (uchar4 *)malloc(sizeof(uchar4)*imageW*imageH);

	clock_t start, stop;
	float t = 0.0;

	/* Start timer */
	assert((start = clock())!=-1);
	for(int i=0; i<imageW; i++)
	{
		for(int j=0; j<imageH; j++)
		{
			index = (imageW)*j+i;
			gray = (src[index].x + src[index].y + src[index].z)/3;
			gray_src[index].x = gray_src[index].y = gray_src[index].z = gray;
		}
	}

	for(int i=1; i<imageW-1; i++)
	{
		for(int j=1; j<imageH-1; j++)
		{
	
			unsigned char pix00 = gray_src[(imageW)*(j-1)+(i-1)].x;
			unsigned char pix01 = gray_src[(imageW)*(j-1)+i].x;
			unsigned char pix02 = gray_src[(imageW)*(j-1)+(i+1)].x;

			unsigned char pix10 = gray_src[(imageW)*j+(i-1)].x;
			unsigned char pix11 = gray_src[(imageW)*j+i].x;
			unsigned char pix12 = gray_src[(imageW)*j+(i+1)].x;

			unsigned char pix20 = gray_src[(imageW)*(j+1)+(i-1)].x;
			unsigned char pix21 = gray_src[(imageW)*(j+1)+i].x;
			unsigned char pix22 = gray_src[(imageW)*(j+1)+(i+1)].x;
		
			pixel = Sobel(
				pix00, pix01, pix02, 
				pix10, pix11, pix12,
				pix20, pix21, pix22);

			//printf("%d\n", pixel);
			dst[(imageW)*j+i] = make_color_uint(pixel, pixel, pixel, 255);

		}
	}

	/* Stop timer */
	stop = clock();
	t = (float) (stop-start)/CLOCKS_PER_SEC;

	free(gray_src);
	return t;
}

float binaryErosionCPU (float4 *src, unsigned int *dst, int imageW, int imageH, int threshold, int iteration, float brightness, float contrast, int mask_w, int mask_h, int adjust){
	
	int index=0;
	unsigned int *bw_src = (unsigned int *)malloc(sizeof(unsigned int)*imageW*imageH);
	float4 fnew = {0, 0, 0, 0};

	clock_t start, stop;
	float t = 0.0;

	/* Start timer */
	assert((start = clock())!=-1);

	binarizeCPU(src, bw_src, imageW, imageH, threshold, brightness, contrast, adjust);

		for(int x=0; x<imageW; x++)
		{
			for(int y=0; y<imageH; y++)
			{		
				int match = 0;
				for (int m = x - mask_w ; m < x + mask_w && !match; m++){
					for (int n = y - mask_h ; n < y + mask_h && !match; n++){
						index = (imageW)*n+m;					
						if(x-mask_w >= 0 && x+mask_w <= imageW && y-mask_h >=0 && y+mask_h <= imageH)
							if (bw_src[index] == make_color_float(1.0, 1.0, 1.0, 1.0))
								match = 1;
					}
				}
				  
				if(!match)
				dst[imageW * y + x] = make_color_float(0, 0, 0, 1.0);
				else
				dst[imageW * y + x] = make_color_float(1.0, 1.0, 1.0, 1.0);
							
			}
		}



	/* Stop timer */
	stop = clock();
	t = (float) (stop-start)/CLOCKS_PER_SEC;

	free(bw_src);

	return t;
}

float binaryDilationCPU (float4 *src, unsigned int *dst, int imageW, int imageH, int threshold, int iteration, float brightness, float contrast, int mask_w, int mask_h, int adjust){
	
	int index=0;
	unsigned int *bw_src = (unsigned int *)malloc(sizeof(unsigned int)*imageW*imageH);
	float4 fnew = {0,0,0,0};

	clock_t start, stop;
	float t = 0.0;

	/* Start timer */
	assert((start = clock())!=-1);

	binarizeCPU(src, bw_src, imageW, imageH, threshold, brightness, contrast, adjust);

		for(int x=0; x<imageW; x++)
		{
			for(int y=0; y<imageH; y++)
			{		

				int match = 1;
				for (int m = x - mask_w ; m < x + mask_w && match; m++){
					for (int n = y - mask_h ; n < y + mask_h && match; n++){
						index = (imageW)*n+m;					
						if(x-mask_w >= 0 && x+mask_w <= imageW && y-mask_h >=0 && y+mask_h <= imageH)
							if (bw_src[index] == make_color_float(0.0, 0.0, 0.0, 0.0))
								match = 0;
					}
				}
				  

				if(match)
					dst[imageW * y + x] = make_color_float(1.0, 1.0, 1.0, 1.0);			
				else
					dst[imageW * y + x] = make_color_float(0, 0, 0, 1.0);
							
			}
		}

	/* Stop timer */
	stop = clock();
	t = (float) (stop-start)/CLOCKS_PER_SEC;

	free(bw_src);

	return t;
}
float grayErosionCPU (float4 *src, unsigned int *dst, int imageW, int imageH, int threshold, int iteration, float brightness, float contrast, int mask_w, int mask_h, int adjust){
	
	int index=0;
	unsigned int *gray_src = (unsigned int *)malloc(sizeof(unsigned int)*imageW*imageH);
	
	clock_t start, stop;
	float t = 0.0;

	/* Start timer */
	assert((start = clock())!=-1);

	grayscaleCPU(src, gray_src, imageW, imageH, brightness, contrast, adjust);

		for(int x=0; x<imageW; x++)
		{
			for(int y=0; y<imageH; y++)
			{		
				unsigned int new_min = 0;
				unsigned int min = gray_src[imageW*y+x];
				for (int m = x - mask_w +1 ; m < x + mask_w -1; m++){
					for (int n = y - mask_h +1 ; n < y + mask_h -1; n++){
						index = (imageW)*n+m;					
						if(x-mask_w >= 0 && x+mask_w <= imageW && y-mask_h >=0 && y+mask_h <= imageH)
							new_min = gray_src[index];
							if (min > new_min)
								min = new_min;
					}
				}
				  
				
				dst[imageW * y + x] = min;
							
			}
		}


	/* Stop timer */
	stop = clock();
	t = (float) (stop-start)/CLOCKS_PER_SEC;

	free(gray_src);

	return t;
}

float grayDilationCPU (float4 *src, unsigned int *dst, int imageW, int imageH, int threshold, int iteration, float brightness, float contrast, int mask_w, int mask_h, int adjust){
	
	int index=0;
	unsigned int *gray_src = (unsigned int *)malloc(sizeof(unsigned int)*imageW*imageH);
	
	clock_t start, stop;
	float t = 0.0;

	/* Start timer */
	assert((start = clock())!=-1);

	grayscaleCPU(src, gray_src, imageW, imageH, brightness, contrast, adjust);

		for(int x=0; x<imageW; x++)
		{
			for(int y=0; y<imageH; y++)
			{		
				unsigned int new_max = 0;
				unsigned int max = gray_src[imageW*y+x];
				for (int m = x - mask_w +1 ; m < x + mask_w -1; m++){
					for (int n = y - mask_h +1 ; n < y + mask_h -1; n++){
						index = (imageW)*n+m;					
						if(x-mask_w >= 0 && x+mask_w <= imageW && y-mask_h >=0 && y+mask_h <= imageH)
							new_max = gray_src[index];
							if (max < new_max)
								max = new_max;
					}
				} 
				
				dst[imageW * y + x] = max;
							
			}
		}


	/* Stop timer */
	stop = clock();
	t = (float) (stop-start)/CLOCKS_PER_SEC;

	free(gray_src);

	return t;
}

float sharpenCPU (float4 *src, unsigned int *dst, int imageW, int imageH, float brightness, float contrast, int adjust)
{
	float pixel_x = 0;
	float pixel_y = 0;
	float pixel_z = 0;

	clock_t start, stop;
	float t = 0.0;

	float pix00; 
	float pix01; 
	float pix02; 

	float pix10;
	float pix11;
	float pix12;

	float pix20;
	float pix21;
	float pix22;

	int index = 0;
	float4 *temp = (float4*)malloc(sizeof(float4)*imageW*imageH);

	/* Start timer */
	assert((start = clock())!=-1);

	if(adjust){
		for(int i=0; i<imageW; i++)
		{
			for(int j=0; j<imageH; j++)
			{
				index = (imageW)*j+i; 
				temp[index] = adjust_contrast_float(src[index], contrast);
				temp[index] = adjust_brightness_float(temp[index], brightness);
			}
		}

		for(int i=1; i<imageW-1; i++)
		{
			for(int j=1; j<imageH-1; j++)
			{
		
				pix00 = temp[(imageW)*(j-1)+(i-1)].x;
				pix01 = temp[(imageW)*(j-1)+i].x;
				pix02 = temp[(imageW)*(j-1)+(i+1)].x;

				pix10 = temp[(imageW)*j+(i-1)].x;
				pix11 = temp[(imageW)*j+i].x;
				pix12 = temp[(imageW)*j+(i+1)].x;

				pix20 = temp[(imageW)*(j+1)+(i-1)].x;
				pix21 = temp[(imageW)*(j+1)+i].x;
				pix22 = temp[(imageW)*(j+1)+(i+1)].x;
			
				pixel_x = Highpass(
					pix00, pix01, pix02, 
					pix10, pix11, pix12,
					pix20, pix21, pix22);

				pix00 = temp[(imageW)*(j-1)+(i-1)].y;
				pix01 = temp[(imageW)*(j-1)+i].y;
				pix02 = temp[(imageW)*(j-1)+(i+1)].y;

				pix10 = temp[(imageW)*j+(i-1)].y;
				pix11 = temp[(imageW)*j+i].y;
				pix12 = temp[(imageW)*j+(i+1)].y;

				pix20 = temp[(imageW)*(j+1)+(i-1)].y;
				pix21 = temp[(imageW)*(j+1)+i].y;
				pix22 = temp[(imageW)*(j+1)+(i+1)].y;
			
				pixel_y = Highpass(
					pix00, pix01, pix02, 
					pix10, pix11, pix12,
					pix20, pix21, pix22);

				pix00 = temp[(imageW)*(j-1)+(i-1)].z;
				pix01 = temp[(imageW)*(j-1)+i].z;
				pix02 = temp[(imageW)*(j-1)+(i+1)].z;

				pix10 = temp[(imageW)*j+(i-1)].z;
				pix11 = temp[(imageW)*j+i].z;
				pix12 = temp[(imageW)*j+(i+1)].z;

				pix20 = temp[(imageW)*(j+1)+(i-1)].z;
				pix21 = temp[(imageW)*(j+1)+i].z;
				pix22 = temp[(imageW)*(j+1)+(i+1)].z;
			
				pixel_z = Highpass(
					pix00, pix01, pix02, 
					pix10, pix11, pix12,
					pix20, pix21, pix22);

				dst[(imageW)*j+i] = make_color_float(pixel_x, pixel_y, pixel_z, 1.f);

			}
		}

	}else{

		for(int i=1; i<imageW-1; i++)
		{
			for(int j=1; j<imageH-1; j++)
			{
		
				pix00 = src[(imageW)*(j-1)+(i-1)].x;
				pix01 = src[(imageW)*(j-1)+i].x;
				pix02 = src[(imageW)*(j-1)+(i+1)].x;

				pix10 = src[(imageW)*j+(i-1)].x;
				pix11 = src[(imageW)*j+i].x;
				pix12 = src[(imageW)*j+(i+1)].x;

				pix20 = src[(imageW)*(j+1)+(i-1)].x;
				pix21 = src[(imageW)*(j+1)+i].x;
				pix22 = src[(imageW)*(j+1)+(i+1)].x;
			
				pixel_x = Highpass(
					pix00, pix01, pix02, 
					pix10, pix11, pix12,
					pix20, pix21, pix22);

				pix00 = src[(imageW)*(j-1)+(i-1)].y;
				pix01 = src[(imageW)*(j-1)+i].y;
				pix02 = src[(imageW)*(j-1)+(i+1)].y;

				pix10 = src[(imageW)*j+(i-1)].y;
				pix11 = src[(imageW)*j+i].y;
				pix12 = src[(imageW)*j+(i+1)].y;

				pix20 = src[(imageW)*(j+1)+(i-1)].y;
				pix21 = src[(imageW)*(j+1)+i].y;
				pix22 = src[(imageW)*(j+1)+(i+1)].y;
			
				pixel_y = Highpass(
					pix00, pix01, pix02, 
					pix10, pix11, pix12,
					pix20, pix21, pix22);

				pix00 = src[(imageW)*(j-1)+(i-1)].z;
				pix01 = src[(imageW)*(j-1)+i].z;
				pix02 = src[(imageW)*(j-1)+(i+1)].z;

				pix10 = src[(imageW)*j+(i-1)].z;
				pix11 = src[(imageW)*j+i].z;
				pix12 = src[(imageW)*j+(i+1)].z;

				pix20 = src[(imageW)*(j+1)+(i-1)].z;
				pix21 = src[(imageW)*(j+1)+i].z;
				pix22 = src[(imageW)*(j+1)+(i+1)].z;
			
				pixel_z = Highpass(
					pix00, pix01, pix02, 
					pix10, pix11, pix12,
					pix20, pix21, pix22);

				dst[(imageW)*j+i] = make_color_float(pixel_x, pixel_y, pixel_z, 1.f);

			}
		}

	}

	
	/* Stop timer */
	stop = clock();
	t = (float) (stop-start)/CLOCKS_PER_SEC;

	return t;
}