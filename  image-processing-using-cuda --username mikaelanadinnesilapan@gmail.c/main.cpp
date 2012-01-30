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

#include "glui.h"


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
char fps[256];

#define BUFFER_DATA(i) ((char *)0 + i)
char *image_file = "images/hug.png";
char *image_format;
int  processing_Kernel = 0;
uchar4 *h_Src;
int imageW, imageH;
int mean_radius = 1;
int threshold = 150;
float contrast = 1.0;
float brightness = 0.0;


//GLui
int main_window = 0;
GLUI *sub_window;
GLUI_Panel *panel_1, *panel_2, *panel_3;
GLUI_Spinner *s_brightness, *s_contrast, *s_threshold, *s_radius;
GLUI_EditText *counter;
GLUI_Button *binarize, *smoothen;
char runtime[25];
#define _NEWLINE_ sub_window->add_statictext( "" );


/**
Functions
**/


void control_cb( int control )
{		
    processing_Kernel = control;
}


void initGL( int *argc, char **argv )
{
    printf("Initializing GLUT...\n");
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);

	if(imageW > imageH){
		if(imageW>1000)
			glutInitWindowSize(imageW-430, imageH-400);	
		if(imageW>800)
			glutInitWindowSize(imageW-230, imageH-200);
		else
			glutInitWindowSize(imageW+130, imageH+100);	
	}else{
		if(imageH>520)
			glutInitWindowSize(imageW+130, imageH+200);	
	}    

    glutInitWindowPosition( 50, 50 );
    main_window = glutCreateWindow("Image Processing with CUDA");	
	
	sub_window = GLUI_Master.create_glui_subwindow( main_window, GLUI_SUBWINDOW_RIGHT );
	sub_window->set_main_gfx_window( main_window );

	_NEWLINE_
	sub_window->add_statictext( "IMAGE PROCESSING USING CUDA" );
	sub_window->add_statictext( "Mikaela Nadinne A. Silapan" );
	sub_window->add_separator();
	_NEWLINE_

	char str[100];
	sprintf(str, "Loaded image: %s", image_file);
	sub_window->add_statictext(str);
	sprintf(str, "Image Size: %d x %d", imageW, imageH);
	sub_window->add_statictext(str);	
	sub_window->add_button( "Press to View Image", 0, control_cb );
	_NEWLINE_

	panel_1  = sub_window->add_panel( "" );
	s_brightness  = sub_window->add_spinner_to_panel( panel_1, "Brightness:", GLUI_SPINNER_FLOAT, &brightness);
	s_brightness->set_int_limits( 0.1f, 1.0f );
	s_brightness->set_alignment( GLUI_ALIGN_RIGHT );

	s_contrast  = sub_window->add_spinner_to_panel( panel_1, "Contrast:", GLUI_SPINNER_FLOAT, &contrast);
	s_contrast->set_int_limits( 0.2f, 5.0f );
	s_contrast->set_alignment( GLUI_ALIGN_RIGHT );

	_NEWLINE_
	sub_window->add_button( "Grayscale", 1, control_cb );
	sub_window->add_button( "Invert", 2, control_cb );
	
	panel_2  = sub_window->add_panel( "" );
	sub_window->add_button_to_panel( panel_2, "Binarize", 3, control_cb);
	s_threshold  = sub_window->add_spinner_to_panel(panel_2, "Threshold:", GLUI_SPINNER_INT, &threshold);
	s_threshold->set_int_limits( 10, 255 );
	s_threshold->set_alignment( GLUI_ALIGN_RIGHT );
	s_threshold->disable();

	panel_3  = sub_window->add_panel( "" );
	sub_window->add_button_to_panel( panel_3, "Smoothen", 4, control_cb );
	s_radius  = sub_window->add_spinner_to_panel(panel_3, "Radius", GLUI_SPINNER_INT, &mean_radius);
	s_radius->set_int_limits( 1, 5 );
	s_radius->set_alignment( GLUI_ALIGN_RIGHT );
	s_radius->disable();

	sub_window->add_button( "Detect Edges", 5, control_cb );
	sub_window->add_button( "Sharpen", 10, control_cb );

	_NEWLINE_
	counter =  sub_window->add_edittext( "FPS: ", GLUI_EDITTEXT_INT, &fps );

	_NEWLINE_	
	sub_window->add_button( "Quit", 0, (GLUI_Update_CB)exit );

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
   		float ifps = 1.f / (cutGetAverageTimerValue(hTimer) / 1000.f);
        sprintf(fps, "%3.1f", ifps);

		//counter->set_text(fps);
        fpsCount = 0; 

        cutilCheckError(cutResetTimer(hTimer));        
    }
}

void callProcessingKernel(uint *dst)
{

	switch(processing_Kernel)
	{
		case 0:
			copyImageWrapper(dst, imageW, imageH, brightness, contrast);
			glutSetWindowTitle("Original Image");
			s_threshold->disable();		
			s_radius->disable();
			break;
		case 1:
			grayImageWrapper(dst, imageW, imageH, brightness, contrast);
			glutSetWindowTitle("Grayscaled Image");
			s_threshold->disable();		
			s_radius->disable();
			break;
		case 2:
			invertWrapper (dst, imageW, imageH, brightness, contrast);
			glutSetWindowTitle("Inverted Image");
			s_threshold->disable();		
			s_radius->disable();
			break;
		case 3:
			binarizationWrapper(dst, imageW, imageH, threshold, brightness, contrast);	
			glutSetWindowTitle("Binarized Image");
			s_threshold->enable();
			s_radius->disable();
			break;
		case 4:
			meanFilterWrapper(dst, imageW, imageH, mean_radius, brightness, contrast);			
			glutSetWindowTitle("Smoothed Image");
			s_threshold->disable();		
			s_radius->enable();
			break;
		case 5:			
			sobelFilterWrapper(dst, imageW, imageH, brightness, contrast);
			glutSetWindowTitle("Sobel Filtered Image");
			s_threshold->disable();		
			s_radius->disable();
			break;
		case 10:
			highPassFilterWrapper(dst, imageW, imageH, brightness, contrast);
			glutSetWindowTitle("Sharpened Image");
			s_threshold->disable();		
			s_radius->disable();
			break;
	}
}
void reshape( int x, int y )
{
  
  int tx, ty, tw, th;
  GLUI_Master.get_viewport_area( &tx, &ty, &tw, &th );
  glViewport( tx, ty, tw, th );

  glutPostRedisplay();
  
}
void display(void){

	cutStartTimer(hTimer);
    uint *d_dst = NULL;
	size_t num_bytes;

    if(frameCounter++ == 0) cutResetTimer(hTimer);
    
	cutilSafeCall(cudaGraphicsMapResources(1, &cuda_pbo_resource, 0));
	cutilCheckMsg("cudaGraphicsMapResources failed");
    cutilSafeCall(cudaGraphicsResourceGetMappedPointer((void**)&d_dst, &num_bytes, cuda_pbo_resource));
	cutilCheckMsg("cudaGraphicsResourceGetMappedPointer failed");
    cutilSafeCall( CUDA_BindTextureToArray() );

	//call function for launching the kernels	
	callProcessingKernel(d_dst);

    cutilSafeCall( CUDA_UnbindTexture() );	
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

	cutStopTimer(hTimer);
	computeFPS();

	glutSwapBuffers();
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
	}
     
}




int main(int argc, char **argv)
{
	// First load the image, so we know what the size of the image (imageW and imageH)    
	//image data is now in h_Src
	loadImageSDL(&h_Src, &imageW, &imageH, image_file); 	
		
	// First initialize OpenGL context, create opengl window
	initGL( &argc, argv );

	//get device with max gflops and use it as device for setting GL
    cudaGLSetGLDevice( cutGetMaxGflopsDeviceId() );
    cutilSafeCall( CUDA_MallocArray(&h_Src, imageW, imageH) );

	initOpenGLBuffers();

	/*
	threshold
	brightness, contrast, 
	smoothing, edge detection,
	invert, 


	enhancement
	erosion and dilation
	opening and closing
	*/
	
    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
	GLUI_Master.set_glutReshapeFunc(reshape);  

    cutilCheckError( cutCreateTimer(&hTimer) );
    cutilCheckError( cutStartTimer(hTimer)   );

    glutMainLoop();	


	//at exit

	printf("Cleaning up...\n");
    cutilCheckError( cutStopTimer(hTimer)   );
    cutilCheckError( cutDeleteTimer(hTimer) );
	cutilSafeCall(cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, gl_PBO, 
						   cudaGraphicsMapFlagsWriteDiscard));
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
    glDeleteBuffers(1, &gl_PBO);
    glDeleteTextures(1, &gl_Tex);

    cutilSafeCall( CUDA_FreeArray() );
    free(h_Src);
	printf("Shutdown done.\n");

    cutilExit(argc, argv);
    cudaThreadExit();

}
