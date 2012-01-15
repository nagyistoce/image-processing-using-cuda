#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <GL/glew.h>
#include <cuda_runtime.h>
#include <cutil_inline.h>
#include <cutil_gl_inline.h>
#include <cutil_gl_error.h>
#include <cuda_gl_interop.h>
#include <GL/glut.h>

#include "cudafunctions.h"
#include <rendercheck_gl.h>



/**
	Global Declarations
**/

GLuint gl_PBO, gl_Tex;
GLuint shader;
struct cudaGraphicsResource *cuda_pbo_resource; // handles OpenGL-CUDA exchange


bool    g_FPS = false;
unsigned int hTimer;
const int frameN = 24;
int frameCounter = 0;
int fpsCount = 0;        // FPS count for averaging
int fpsLimit = 1;        // FPS limit for sampling
unsigned int frameCount = 0;


#define BUFFER_DATA(i) ((char *)0 + i)
int  processing_Kernel = 0;
uchar4 *h_Src;
int imageW, imageH;
int mean_radius = 8;
int radius_step = 2;


/**
	Functions
**/

void initGL( int *argc, char **argv )
{
    printf("Initializing GLUT...\n");
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize(imageW, imageH);
    glutInitWindowPosition(512 - imageW / 2, 384 - imageH / 2);
    glutCreateWindow(argv[0]);
    printf("OpenGL window created.\n");

    glewInit();
}

// shader for displaying floating-point texture
static const char *shader_code = 
"!!ARBfp1.0\n"
"TEX result.color, fragment.texcoord, texture[0], 2D; \n"
"END";

GLuint compileASMShader(GLenum program_type, const char *code)
{
    GLuint program_id;
    glGenProgramsARB(1, &program_id);
    glBindProgramARB(program_type, program_id);
    glProgramStringARB(program_type, GL_PROGRAM_FORMAT_ASCII_ARB, (GLsizei) strlen(code), (GLubyte *) code);

    GLint error_pos;
    glGetIntegerv(GL_PROGRAM_ERROR_POSITION_ARB, &error_pos);
    if (error_pos != -1) {
        const GLubyte *error_string;
        error_string = glGetString(GL_PROGRAM_ERROR_STRING_ARB);
        fprintf(stderr, "Program error at position: %d\n%s\n", (int)error_pos, error_string);
        return 0;
    }
    return program_id;
}

void initOpenGLBuffers()
{
    printf("Creating GL texture...\n");
        glEnable(GL_TEXTURE_2D);
		//allocate a texture, give it a name to reference to :P
        glGenTextures(1, &gl_Tex);
		//select current texture
        glBindTexture(GL_TEXTURE_2D, gl_Tex);
		//clamp to the edges if lumampas. :P
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
		//when texture area is large, do the same thing
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		//when texture area selected is small, get average of chuhchuhu
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, imageW, imageH, 0, GL_RGBA, GL_UNSIGNED_BYTE, h_Src);
    printf("Texture created.\n");

    printf("Creating PBO...\n");
        glGenBuffers(1, &gl_PBO);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, gl_PBO);
        glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, imageW * imageH * 4, h_Src, GL_STREAM_COPY);
        //While a PBO is registered to CUDA, it can't be used 
        //as the destination for OpenGL drawing calls.
        //But in our particular case OpenGL is only used 
        //to display the content of the PBO, specified by CUDA kernels,
        //so we need to register/unregister it only once.
	// DEPRECATED: cutilSafeCall( cudaGLRegisterBufferObject(gl_PBO) );
    cutilSafeCall(cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, gl_PBO, 
					       cudaGraphicsMapFlagsWriteDiscard));
        CUT_CHECK_ERROR_GL();
    printf("PBO created.\n");

    // load shader program
    shader = compileASMShader(GL_FRAGMENT_PROGRAM_ARB, shader_code);

}

void computeFPS()
{
    frameCount++;
    fpsCount++;
    
	if (fpsCount == fpsLimit) {        
   		char fps[256];
   		float ifps = 1.f / (cutGetAverageTimerValue(hTimer) / 1000.f);
        sprintf(fps, "%3.1f fps", ifps);

        glutSetWindowTitle(fps);
        fpsCount = 0; 

        cutilCheckError(cutResetTimer(hTimer));        
    }
}

void callProcessingKernel(uint *dst)
{
	switch(processing_Kernel)
	{
		case 1:
			copyImageWrapper(dst, imageW, imageH);
			break;
		case 2:
			grayImageWrapper(dst, imageW, imageH);
			break;
		case 3:
			meanFilterWrapper(dst, imageW, imageH, mean_radius);
			break;
		case 4:
			medianFilterWrapper(dst, imageW, imageH);
			break;
		case 5:			
			sobelFilterWrapper(dst, imageW, imageH);
			break;
	}
}
void display(void){

	cutStartTimer(hTimer);
    uint *d_dst = NULL;
	size_t num_bytes;

    if(frameCounter++ == 0) cutResetTimer(hTimer);
    // DEPRECATED: cutilSafeCall(cudaGLMapBufferObject((void**)&d_dst, gl_PBO));
    cutilSafeCall(cudaGraphicsMapResources(1, &cuda_pbo_resource, 0));
	cutilCheckMsg("cudaGraphicsMapResources failed");
    cutilSafeCall(cudaGraphicsResourceGetMappedPointer((void**)&d_dst, &num_bytes, cuda_pbo_resource));
	cutilCheckMsg("cudaGraphicsResourceGetMappedPointer failed");

    cutilSafeCall( CUDA_BindTextureToArray() );

	//call function for launching the kernels
	callProcessingKernel(d_dst);

    cutilSafeCall( CUDA_UnbindTexture() );
    
	// DEPRECATED: cutilSafeCall(cudaGLUnmapBufferObject(gl_PBO));
	cutilSafeCall(cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0));

	{
        glClear(GL_COLOR_BUFFER_BIT);

        glTexSubImage2D( GL_TEXTURE_2D, 0, 0, 0, imageW, imageH, GL_RGBA, GL_UNSIGNED_BYTE, BUFFER_DATA(0) );
        glBegin( GL_QUADS );
		  glTexCoord2d(0.0,0.0); glVertex2d(+1.0,+1.0);
		  glTexCoord2d(1.0,0.0); glVertex2d(-1.0,+1.0);
		  glTexCoord2d(1.0,1.0); glVertex2d(-1.0,-1.0);
		  glTexCoord2d(0.0,1.0); glVertex2d(+1.0,-1.0);
		glEnd();
        glFinish();
    }

    if(frameCounter == frameN){
        frameCounter = 0;
        if(g_FPS){
            printf("FPS: %3.1f\n", frameN / (cutGetTimerValue(hTimer) * 0.001) );
            g_FPS = false;
        }
    }

	glutSwapBuffers();

	cutStopTimer(hTimer);
	computeFPS();

	glutPostRedisplay();
}



void keyboard(unsigned char k, int /*x*/, int /*y*/)
{
    switch (k){
        case '\033':
        case 'q':
        case 'Q':
            printf("Cleaning up...\n");
                cutilCheckError( cutStopTimer(hTimer)   );
                cutilCheckError( cutDeleteTimer(hTimer) );
				// DEPRECATED: cutilSafeCall( cudaGLRegisterBufferObject(gl_PBO) );
				cutilSafeCall(cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, gl_PBO, 
									   cudaGraphicsMapFlagsWriteDiscard));
                glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
                glDeleteBuffers(1, &gl_PBO);
                glDeleteTextures(1, &gl_Tex);

                cutilSafeCall( CUDA_FreeArray() );
                free(h_Src);
            printf("Shutdown done.\n");
            cudaThreadExit();
            exit(0);
        break;

		case '1':
			printf("Original image.\n");
			processing_Kernel = 1;
		break;

		case '2':
			printf("Grayscale image.\n");
			processing_Kernel = 2;
		break;

		case '3':
			printf("Mean filtering applied.\n");
			processing_Kernel = 3;
		break;

		case '4':
			printf("Median filtering applied.\n");
			processing_Kernel = 4;
		break;

		case '5':			
			processing_Kernel = 5;
		break;

		case '6':			
			processing_Kernel = 6;
		break;

		case '7':			
			processing_Kernel = 7;
		break;

		case '-':
			if (processing_Kernel == 3) {
				mean_radius -= radius_step;
				if (mean_radius < MIN_MEAN_RADIUS) mean_radius = MIN_MEAN_RADIUS;
				printf("Windows Radius = %d\n", mean_radius);
			}
		break;

		case '=':			
			if (processing_Kernel == 3) {
				mean_radius += radius_step;
				if (mean_radius > MAX_MEAN_RADIUS(imageW)) mean_radius = MAX_MEAN_RADIUS(imageW);
				printf("Window Radius = %d\n", mean_radius);
			}
		break;
	}
     
}














/*
**	Main Program
**	Summary:
**	Check Debug Mode
**	Check if device is capable of rendering in OpenGL mode
**	if yes, load image data
**	initGL
**	allocate device memory: h_Src = a_Src [in CUDA code so it's not declared here]
**	init openGL buffers for rendering.. shader program compilation.. reg and unreg PBO
*/

int main(int argc, char **argv)
{

	// First load the image, so we know what the size of the image (imageW and imageH)    
	//image data is now in h_Src
	loadImageSDL(&h_Src, &imageW, &imageH, "images/piggies.jpg"); 	
		
	// First initialize OpenGL context, create opengl window
	initGL( &argc, argv );

	//get device with max gflops and use it as device for setting GL
    cudaGLSetGLDevice( cutGetMaxGflopsDeviceId() );
    cutilSafeCall( CUDA_MallocArray(&h_Src, imageW, imageH) );

	initOpenGLBuffers();
		
	printf("\t%s\n", argv[0]);
    printf("Press [1] to view original image\n");
    printf("Press [2] to view image in grayscale\n");
	printf("Press [3] to view image using mean filtering\n");
	printf("\tPress [-] to decrease window radius\n\tPress [=] to increase window radius\n");
	printf("Press [4] to view image using median filtering\n");
    printf("Press [q] to exit\n");

	glutIdleFunc(display);
    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
    cutilCheckError( cutCreateTimer(&hTimer) );
    cutilCheckError( cutStartTimer(hTimer)   );
    glutMainLoop();
	

    cutilExit(argc, argv);
    cudaThreadExit();

}
