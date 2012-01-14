#include "SDL.h"
#include "SDL_image.h"

typedef struct {
	unsigned char x, y, z, w;
} uchar4;

Uint32 getpixel(SDL_Surface *surface, int x, int y) 
{
	int bpp = surface->format->BytesPerPixel;

	/* Here p is the address to the pixel we want to retrieve */
	Uint8 *p = (Uint8 *)surface->pixels + y * surface->pitch + x * bpp;

	switch (bpp) {

	case 1:
	  return *p;

	case 2:
	   return *(Uint16 *)p;

	case 3:
	   if (SDL_BYTEORDER == SDL_BIG_ENDIAN)
		   return p[0] << 16 | p[1] << 8 | p[2];
	   else
		   return p[0] | p[1] << 8 | p[2] << 16;

	case 4:
	   return *(Uint32 *)p;

	default:
	   return 0;       /* shouldn't happen, but avoids warnings */
	} // switch

}



SDL_Surface* load_Image(const char* file)
{
	//The image that's loaded
    SDL_Surface* loadedImage = NULL;

    //Load the image
    loadedImage = IMG_Load(file);

    //Return the image
    return loadedImage;
}


extern "C" void loadImageSDL ( uchar4 **dst, int *w, int *h, char *file ) {
	
	SDL_Surface *image = load_Image(file);
	if(!image) {
		printf("Image file not loaded successfully! Press enter to exit...\n");
		getchar();
		exit(-1);
	}

	*dst = (uchar4 *)malloc(sizeof(uchar4)*image->w*image->h);
	*w = image->w;
	*h = image->h;

	size_t len = strlen(file);
	char *image_format = (char*)malloc(len);
	image_format[len] = '\0';
	strncpy(image_format, (file+(len-3)), 4);
	
	printf("Loaded image format: %s...\n", image_format);
	int notBMP = strcmp(image_format, "bmp");

	unsigned int pixel = 0;

	SDL_LockSurface( image );

	if (notBMP){		
		//copy from surface to unsigned int destination array
		for(int x=0; x<*w; x++){
			for(int y=0; y<*h; y++){
				pixel = getpixel(image, x, y);		
				(*dst)[(*w)*y+x].x = (pixel) & 0xff;
				(*dst)[(*w)*y+x].y = (pixel>>8) & 0xff;
				(*dst)[(*w)*y+x].z = (pixel>>16) & 0xff;
			}
		}
	}else{ //change order of colors. GRB in bmp images. :)
		//copy from surface to unsigned int destination array
		for(int x=0; x<*w; x++){
			for(int y=0; y<*h; y++){
				pixel = getpixel(image, x, y);		
				(*dst)[(*w)*y+x].x = (pixel>>16) & 0xff;
				(*dst)[(*w)*y+x].y = (pixel>>8) & 0xff;
				(*dst)[(*w)*y+x].z = (pixel) & 0xff;
			}
		}
	}

	printf("Loaded image: %s\n", file);
	
	SDL_UnlockSurface( image );
	SDL_FreeSurface(image);
	SDL_Quit();
} 