
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <GL/glew.h>
#include <GL/glut.h>
#include <cuda_runtime.h>
#include <cutil_inline.h>
#include <cutil_gl_inline.h>
#include <cutil_gl_error.h>
#include <cuda_gl_interop.h>

#include <GL/glui.h>
#include <rendercheck_gl.h>

#include "cpufunctions.h"
#include "gpufunctions.h"

/**
	Global Declarations
**/

GLuint gl_PBO, gl_Tex;
GLuint shader;
struct cudaGraphicsResource *cuda_pbo_resource; // handles OpenGL-CUDA exchange

#define BUFFER_DATA(i) ((char *)0 + i)
const char *image_file = "images/barkadapic.jpg"; //sample 5 higher results to an error
std::string loaded_file_name;
char *image_format;
int  processing_Kernel = 1;

uchar4 *h_Src = NULL;
float4 *h_FSrc = NULL;
unsigned int *d_temp = NULL;

int imageW, imageH;
int threshold = 150;
float contrast = 1.0;
float brightness = 0.0;
int processor = 0;
float cycles = 0.0f;
float runtime = 0.f;
float t = 0.0f;

int box_radius = 1;
int iteration = 1;
int mask_radius = 5;
int mask_w = mask_radius/2;
int mask_h = mask_radius/2;

/*
	GLui - User Interface
*/
int main_window = 0;
bool loaded = false;
char str[100];
GLUI *window_menu, *window_popup;
GLUI_Panel *panel_filebrowser, *panel_kernels, *panel_enhancements, *panel_process, *panel_image;
GLUI_Spinner *s_brightness, *s_contrast, *s_threshold, *s_mask_radius, *s_iteration, *s_box_radius;
GLUI_Button *copyimage, *grayscale, *invert, *binarize, *smoothen, *edge;
GLUI_EditText *runtime_counter;
GLUI_FileBrowser *file_browser;
GLUI_RadioGroup *radio_group;

// constants
#define VIEW_IMAGE 1
#define GRAYSCALE 2
#define INVERT 3
#define BINARIZE 4
#define SMOOTHEN 5
#define EDGE_DETECT 6
#define B_ERODE 7
#define B_DILATE 8
#define G_ERODE 9
#define G_DILATE 10
#define SHARPEN 11

void control_cb ( int control );
void initGL(int *argc, char **argv);
GLuint compileASMShader(GLenum program_type, const char *code);
void initOpenGLBuffers();
void runByCPU(unsigned int *dst);
void runByGPU(unsigned int *dst);
void idle();
void reshape(int x, int y);
void display();

/*
	FUNCTIONS
*/

void control_cb2( int control )
{		
	if(control == 100){
		loaded_file_name = "";
		loaded_file_name = file_browser->get_file();
		//check if the file is an image
		//get the file extension
		//int index = loaded_file_name.find_last_of("/", loaded_file_name.length());
		//printf("Last Index of /: %d\n", index);
		loaded = true;
	}else if(control == 400){
		mask_w = mask_radius/2;
		mask_h = mask_radius/2;
	}
}

void control_cb( int control )
{
	processing_Kernel = control;
}

void initGL( int *argc, char **argv )
{
    //printf("Initializing GLUT...\n");
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);

	
	//screen resolution size 1366 x 768
	//get aspect ratio and maintain it when resizing images.
	int screenW = 1366;
	int screenH = 768;
	int temp = 0, counter = 1;

	float ratio = 0.f;

	if(imageW > screenW || imageH > screenH) {
		if(imageW > imageH) {
			ratio = imageW/imageH;

			do{
				temp = (int)imageW/(ratio*counter);
				counter++;
			}while(temp>screenW-250);
			
			glutInitWindowSize((int)imageW/(ratio*counter), (int)imageH/(ratio*counter));	

		}
		else {
			ratio = imageH/imageW;
			
			do{
				temp = (int)imageH/(ratio*counter);
				counter++;
			}while(temp>screenH-250);
			
			glutInitWindowSize((int)imageW/(ratio*counter), (int)imageH/(ratio*counter));	
		}
	}else{
		glutInitWindowSize(imageW, imageH);	
	}

	glutInitWindowPosition( 350, 50 );
    main_window = glutCreateWindow("");
	
	window_menu = GLUI_Master.create_glui("", 0, 50, 50);
	
	panel_image = new GLUI_Panel(window_menu, "");
	sprintf(str, "Loaded image: %s", image_file);
	window_menu->add_statictext_to_panel(panel_image, str);
	sprintf(str, "Image size: %d x %d", imageW, imageH);
	window_menu->add_statictext_to_panel(panel_image, str);

	panel_filebrowser = new GLUI_Rollout(window_menu, "Browse for image file:", false);
	file_browser = new GLUI_FileBrowser(panel_filebrowser, "", false, 100, control_cb2);
				
	panel_enhancements = new GLUI_Rollout(window_menu, "Image Enhancements", true);

	//add the spinners	
	s_brightness = window_menu->add_spinner_to_panel(panel_enhancements, "Brightness", GLUI_SPINNER_FLOAT, &brightness);
	s_contrast = window_menu->add_spinner_to_panel(panel_enhancements, "Contrast", GLUI_SPINNER_FLOAT, &contrast);
	s_threshold = window_menu->add_spinner_to_panel(panel_enhancements, "Threshold", GLUI_SPINNER_INT, &threshold);
	s_mask_radius = window_menu->add_spinner_to_panel(panel_enhancements, "Mask Radius", GLUI_SPINNER_INT, &mask_radius, 400, control_cb2);
	s_iteration = window_menu->add_spinner_to_panel(panel_enhancements, "Iteration", GLUI_SPINNER_INT, &iteration);
	s_box_radius = window_menu->add_spinner_to_panel(panel_enhancements, "Box Radius", GLUI_SPINNER_INT, &box_radius);
	//add radius of mask for erosion and dilation


	//set the limits of each spinner
	s_brightness->set_float_limits(0.1f, 1.0f);
	s_contrast->set_float_limits(0.2f, 5.0f);
	s_threshold->set_int_limits(10, 255);
	s_mask_radius->set_float_limits(5, 15);
	s_iteration->set_float_limits(1, 7);
	s_box_radius->set_float_limits(1, 6);
	
	//add the buttons
	panel_kernels = new GLUI_Rollout(window_menu, "Image Processing Kernels", true);
	window_menu->add_button_to_panel(panel_kernels, "View Image", VIEW_IMAGE, control_cb);
	window_menu->add_button_to_panel(panel_kernels, "Grayscale", GRAYSCALE, control_cb);
	window_menu->add_button_to_panel(panel_kernels, "Invert", INVERT, control_cb);
	window_menu->add_button_to_panel(panel_kernels, "Binarize", BINARIZE, control_cb);
	window_menu->add_button_to_panel(panel_kernels, "Smoothen", SMOOTHEN, control_cb);
	window_menu->add_button_to_panel(panel_kernels, "Detect Edge", EDGE_DETECT, control_cb);
	window_menu->add_button_to_panel(panel_kernels, "Binary Erosion", B_ERODE, control_cb);
	window_menu->add_button_to_panel(panel_kernels, "Binary Dilation", B_DILATE, control_cb);
	window_menu->add_button_to_panel(panel_kernels, "Gray Erosion", G_ERODE, control_cb);
	window_menu->add_button_to_panel(panel_kernels, "Gray Dilation", G_DILATE, control_cb);
	window_menu->add_button_to_panel(panel_kernels, "Sharpen", SHARPEN, control_cb);

	panel_process = new GLUI_Rollout(window_menu, "Processor", true);
	radio_group = window_menu->add_radiogroup_to_panel(panel_process, &processor, 200, control_cb2);	
	window_menu->add_radiobutton_to_group(radio_group, "Run by GPU"); //processor is 0
	window_menu->add_radiobutton_to_group(radio_group, "Run by CPU"); //processor is 1
	runtime_counter = window_menu->add_edittext_to_panel(panel_process, "Runtime (seconds): ", GLUI_EDITTEXT_FLOAT, &cycles );
	
	window_menu->add_button("QUIT", 0, (GLUI_Update_CB)exit);
	
	//don't forget glewInit!!!!!!!!
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
   //printf("Creating GL texture...\n");
        glEnable(GL_TEXTURE_2D); 
		//allocate a texture, give it a name to reference to :P
        glGenTextures(1, &gl_Tex); //
		//select current texture
        glBindTexture(GL_TEXTURE_2D, gl_Tex); //
		//clamp to the edges if lumampas. :P
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
		//when texture area is large, do the same thing
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST); //
		//when texture area selected is small, get average of chuhchuhu
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST); //
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, imageW, imageH, 0, GL_RGBA, GL_UNSIGNED_BYTE, h_Src);
    //printf("Texture created.\n");

    //printf("Creating PBO...\n");
        glGenBuffers(1, &gl_PBO); //
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, gl_PBO); //
		//put image data into buffer
        glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, imageW * imageH * 4, h_Src, GL_STREAM_DRAW_ARB); //			
		glRotatef(180,0.0,0.0,1.0); //rotate kasi baligtad :P
        //While a PBO is registered to CUDA, it can't be used 
        //as the destination for OpenGL drawing calls.
        //But in our particular case OpenGL is only used 
        //to display the content of the PBO, specified by CUDA kernels,
        //so we need to register/unregister it only once.
    cutilSafeCall(cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, gl_PBO, 
					       cudaGraphicsMapFlagsWriteDiscard));
    CUT_CHECK_ERROR_GL();
    //printf("PBO created.\n");
	
    // load shader program
    shader = compileASMShader(GL_FRAGMENT_PROGRAM_ARB, shader_code);

}

void runByCPU(unsigned int *dst)
{
	dst = (unsigned int*)malloc(sizeof(unsigned int)*imageH*imageW);	

	switch(processing_Kernel)
	{
		case 1:	
			t = copyImageCPU(h_FSrc, dst, imageW, imageH, brightness, contrast, 1);
			glutSetWindowTitle("Original Image - CPU");
			s_threshold->disable();		
			s_mask_radius->disable();
			s_iteration->disable();
			s_box_radius->disable();
			break;
		case 2:
			t = grayscaleCPU(h_FSrc, dst, imageW, imageH, brightness, contrast, 1);
			glutSetWindowTitle("Grayscaled Image - CPU");
			s_threshold->disable();		
			s_mask_radius->disable();
			s_iteration->disable();
			s_box_radius->disable();
			break;
		case 3:
			t = invertCPU(h_FSrc, dst, imageW, imageH, brightness, contrast, 1);
			glutSetWindowTitle("Inverted Image - CPU");
			s_threshold->disable();
			s_mask_radius->disable();
			s_iteration->disable();
			s_box_radius->disable();
			break;
		case 4:
			t = binarizeCPU(h_FSrc, dst, imageW, imageH, threshold, brightness, contrast, 1);	
			glutSetWindowTitle("Binarized Image - CPU");
			s_threshold->enable();
			s_mask_radius->disable();
			s_iteration->disable();
			s_box_radius->disable();
			break;
		case 5:
			t = smoothenCPU(h_FSrc, dst, imageW, imageH, box_radius, iteration, brightness, contrast, 1);	
			glutSetWindowTitle("Smoothened Image - CPU");
			s_threshold->disable();
			s_mask_radius->disable();
			s_iteration->enable();
			s_box_radius->enable();
			break;
		case 6:
			t = edgeDetectCPU(h_Src, dst, imageW, imageH, brightness, contrast, 1);	
			glutSetWindowTitle("Sobel Filtered Image - CPU");
			s_threshold->disable();
			s_mask_radius->disable();
			s_iteration->disable();
			s_box_radius->disable();
			break;
		case 7:
			t = binaryErosionCPU(h_FSrc, dst, imageW, imageH, threshold, iteration, brightness, contrast, mask_w, mask_h, 1);	
			glutSetWindowTitle("Eroded Binary Image - CPU");
			s_threshold->enable();
			s_mask_radius->enable();
			s_iteration->enable();
			s_box_radius->disable();
			break;
		case 8:
			t = binaryDilationCPU(h_FSrc, dst, imageW, imageH, threshold, iteration, brightness, contrast, mask_w, mask_h, 1);	
			glutSetWindowTitle("Dilated Binary Image - CPU");
			s_threshold->enable();
			s_mask_radius->enable();
			s_iteration->enable();
			s_box_radius->disable();
			break;
		case 9:
			t = grayErosionCPU(h_FSrc, dst, imageW, imageH, threshold, iteration, brightness, contrast, mask_w, mask_h, 1);	
			glutSetWindowTitle("Eroded Gray Image - CPU");
			s_threshold->disable();
			s_mask_radius->enable();
			s_iteration->enable();
			s_box_radius->disable();
			break;
		case 10:
			t = grayDilationCPU(h_FSrc, dst, imageW, imageH, threshold, iteration, brightness, contrast, mask_w, mask_h, 1);	
			glutSetWindowTitle("Dilated Gray Image - CPU");
			s_threshold->disable();
			s_mask_radius->enable();
			s_iteration->enable();
			s_box_radius->disable();
			break;
		case 11:
			t = sharpenCPU(h_FSrc, dst, imageW, imageH, iteration, brightness, contrast, 1);	
			glutSetWindowTitle("Sharpened Image - CPU");
			s_threshold->disable();
			s_mask_radius->disable();
			s_iteration->enable();
			s_box_radius->disable();
			break;

	}

	sprintf(str, "%f", t);
	runtime_counter->set_text(str);

	//put image data into buffer
    glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, imageW * imageH * 4, dst, GL_STREAM_DRAW_ARB);			
	
	free(dst);
}

void runByGPU(unsigned int *dst)
{
	switch(processing_Kernel)
	{
		case VIEW_IMAGE:
			runtime = copyImageWrapper(dst, imageW, imageH, brightness, contrast, 1);
			glutSetWindowTitle("Original Image");
			s_threshold->disable();
			s_mask_radius->disable();
			s_iteration->disable();
			s_box_radius->disable();
			break;
			
		case GRAYSCALE:
			runtime = grayImageWrapper(dst, imageW, imageH, brightness, contrast, 1);
			glutSetWindowTitle("Grayscaled Image");
			s_threshold->disable();
			s_mask_radius->disable();
			s_iteration->disable();
			s_box_radius->disable();
			break;
		case INVERT:
			runtime = invertWrapper (dst, imageW, imageH, brightness, contrast, 1);
			glutSetWindowTitle("Inverted Image");
			s_threshold->disable();
			s_mask_radius->disable();
			s_iteration->disable();
			s_box_radius->disable();
			break;
		case BINARIZE:
			runtime = binarizationWrapper(dst, imageW, imageH, threshold, brightness, contrast, 1);	
			glutSetWindowTitle("Binarized Image");
			s_threshold->enable();
			s_mask_radius->disable();
			s_iteration->disable();
			s_box_radius->disable();
			break;
		case SMOOTHEN:			
			runtime = meanFilterWrapper(dst, imageW, imageH, box_radius, iteration, brightness, contrast, 1);			
			glutSetWindowTitle("Smoothed Image");
			s_threshold->disable();
			s_mask_radius->disable();
			s_iteration->enable();
			s_box_radius->enable();
			break;
		case EDGE_DETECT:			
			runtime = sobelFilterWrapper(dst, imageW, imageH, brightness, contrast, 1);
			glutSetWindowTitle("Sobel Filtered Image");
			s_threshold->disable();
			s_mask_radius->disable();
			s_iteration->disable();
			s_box_radius->disable();
			break;
		case B_ERODE:			
			runtime = binaryErosionWrapper(dst, imageW, imageH, threshold, iteration, brightness, contrast, mask_w, mask_h, 1);			
			glutSetWindowTitle("Eroded Binary Image");
			s_threshold->enable();
			s_mask_radius->enable();
			s_iteration->enable();
			s_box_radius->disable();
			break;
		case B_DILATE:
			runtime = binaryDilationWrapper(dst, imageW, imageH, threshold, iteration, brightness, contrast, mask_w, mask_h, 1);			
			glutSetWindowTitle("Dilated Binary Image");
			s_threshold->enable();	
			s_mask_radius->enable();
			s_iteration->enable();
			s_box_radius->disable();
			break;
		case G_ERODE:	
			runtime = grayErosionWrapper(dst, imageW, imageH, threshold, iteration, brightness, contrast, mask_w, mask_h, 1);			
			glutSetWindowTitle("Eroded Gray Image");
			s_threshold->disable();	
			s_mask_radius->enable();
			s_iteration->enable();
			s_box_radius->disable();
			break;
		case G_DILATE:
			runtime = grayDilationWrapper(dst, imageW, imageH, threshold, iteration, brightness, contrast, mask_w, mask_h, 1);			
			glutSetWindowTitle("Dilated Gray Image");
			s_threshold->disable();	
			s_mask_radius->enable();
			s_iteration->enable();
			s_box_radius->disable();
			break;
		case SHARPEN:
			runtime = highPassFilterWrapper(dst, imageW, imageH, iteration, brightness, contrast, 1);
			glutSetWindowTitle("Sharpened Image");
			s_threshold->disable();	
			s_mask_radius->disable();
			s_iteration->enable();
			s_box_radius->disable();
			break;
		default:
			break;
	}	

	sprintf(str, "%f", runtime);
	runtime_counter->set_text(str);	
	cutilSafeCall(cudaFree(d_temp));
}

/*
	GLUT functions
*/

void idle( void ) 
{
	glutSetWindow(main_window);
	glutPostRedisplay();
}

void reshape( int x, int y )
{
  
  int tx, ty, tw, th;
  GLUI_Master.get_viewport_area( &tx, &ty, &tw, &th );
  glViewport( tx, ty, tw, th );

  glutPostRedisplay();
  
}
void display(void){

	unsigned int *d_dst = NULL;
	size_t num_bytes;

	cutilSafeCall(cudaGraphicsMapResources(1, &cuda_pbo_resource, 0));
	cutilCheckMsg("cudaGraphicsMapResources failed");
	cutilSafeCall(cudaGraphicsResourceGetMappedPointer((void**)&d_dst, &num_bytes, cuda_pbo_resource));
	cutilCheckMsg("cudaGraphicsResourceGetMappedPointer failed");


	if(loaded){		
		loaded = false;
		loadImageSDL(&h_Src, &imageW, &imageH, loaded_file_name.c_str()); 	
		loadImageSDLFloat(&h_FSrc, &imageW, &imageH, loaded_file_name.c_str());
		glutReshapeWindow(imageW, imageH);
		cutilSafeCall( CUDA_MallocArray(&h_Src, imageW, imageH) );				
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, imageW, imageH, 0, GL_RGBA, GL_UNSIGNED_BYTE, h_Src);
		glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, imageW * imageH * 4, h_Src, GL_STREAM_DRAW_ARB);	
	}

	//call function for launching the kernels	
	if (processor == 0)
	{	
		cutilSafeCall(CUDA_BindTexture());	
		runByGPU(d_dst);	
		cutilSafeCall(CUDA_UnbindTexture());				
	}
	else
	{	
		runByCPU(d_dst);
	}


    cutilSafeCall(cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0));
	{
        glClear(GL_COLOR_BUFFER_BIT);		
        glDisable(GL_DEPTH_TEST);
        glTexSubImage2D( GL_TEXTURE_2D, 0, 0, 0, imageW, imageH, GL_RGBA, GL_UNSIGNED_BYTE, BUFFER_DATA(0) );
		glBegin( GL_QUADS );				  		
			glTexCoord2f(0.0, 0.0);          
            glVertex2f(1.0, -1.0);

            glTexCoord2f(1.0, 0);          
            glVertex2f(-1.0, -1.0);

            glTexCoord2f(1.0, 1.0);          
			glVertex2f(-1.0, 1.0);

            glTexCoord2f(0.0, 1.0);          
            glVertex2f(1.0, 1.0);
		glEnd();
        glFinish();
    }

	glutSwapBuffers();		
}


void runBenchmarkGPU() 
{
	unsigned int *d_img;

	// allocate device memory
    cutilSafeCall( cudaMalloc( (void**) &d_img,  (imageW * imageH * sizeof(unsigned int)) ));
    CUDA_BindTexture(); 

	// Start round-trip timer and process iCycles loops on the GPU
    const int iCycles = 10;

	printf("\nRunning a %d cycle benchmark on CPU...\n\n", iCycles);
	printf("\nImage Width: %d Image Height: %d\n\n", imageW, imageH);

	float dProcessingTime = 0.0;
	for (int i = 0; i < iCycles; i++)
    {
        dProcessingTime += brightnessWrapper(d_img, imageW, imageH, brightness);
    }

	// check if kernel execution generated an error and sync host
    cutilCheckMsg("Error: Kernel execution FAILED");
    // Get average computation time
    dProcessingTime /= (float)iCycles;
	
	printf("Processing Time of Brightness Adjustment: %.5f\n\n", dProcessingTime);
	
	dProcessingTime = 0.0;
	for (int i = 0; i < iCycles; i++)
    {
        dProcessingTime += gammaCorrectionWrapper(d_img, imageW, imageH, contrast);
    }

	// check if kernel execution generated an error and sync host 
    cutilCheckMsg("Error: Kernel execution FAILED");
    // Get average computation time
    dProcessingTime /= (float)iCycles;
	
	printf("Processing Time of Gamma Correction: %.5f\n\n", dProcessingTime);
	

	dProcessingTime = 0.0;
	for (int i = 0; i < iCycles; i++)
    {
        dProcessingTime += grayImageWrapper(d_img, imageW, imageH, brightness, contrast, 0);
    }

    // check if kernel execution generated an error and sync host
    cutilCheckMsg("Error: Kernel execution FAILED");
    // Get average computation time
    dProcessingTime /= (float)iCycles;
	
	printf("Processing Time of Grayscale Technique: %.5f\n\n", dProcessingTime);
	
	dProcessingTime = 0.0;
	for (int i = 0; i < iCycles; i++)
    {
        dProcessingTime += invertWrapper(d_img, imageW, imageH, brightness, contrast, 0);
    }

	// check if kernel execution generated an error and sync host
    cutilCheckMsg("Error: Kernel execution FAILED");
    // Get average computation time
    dProcessingTime /= (float)iCycles;

	printf("Processing Time of Invert Technique: %.5f\n\n", dProcessingTime);
	
	dProcessingTime = 0.0;
	for (int i = 0; i < iCycles; i++)
    {
        dProcessingTime += sobelFilterWrapper(d_img, imageW, imageH, brightness, contrast, 0);
    }

	// check if kernel execution generated an error and sync host
    cutilCheckMsg("Error: Kernel execution FAILED");
    // Get average computation time
    dProcessingTime /= (float)iCycles;

	printf("Processing Time of Sobel Filter Technique: %.5f\n\n", dProcessingTime);

	dProcessingTime = 0.0;
	for (int i = 0; i < iCycles; i++)
    {
        dProcessingTime += meanFilterWrapper(d_img, imageW, imageH, box_radius, iteration, brightness, contrast, 0);		
    }

	// check if kernel execution generated an error and sync host
    cutilCheckMsg("Error: Kernel execution FAILED");
    // Get average computation time
    dProcessingTime /= (float)iCycles;

	printf("Processing Time of Mean Filter Technique: %.5f\n\n", dProcessingTime);

	dProcessingTime = 0.0;
	for (int i = 0; i < iCycles; i++)
    {
        dProcessingTime += binaryErosionWrapper(d_img, imageW, imageH, threshold, iteration, brightness, contrast, mask_w, mask_h, 0);
    }

	// check if kernel execution generated an error and sync host
    cutilCheckMsg("Error: Kernel execution FAILED");
    // Get average computation time
    dProcessingTime /= (float)iCycles;

	printf("Processing Time of Binary Erosion Technique: %.5f\n\n", dProcessingTime);

	dProcessingTime = 0.0;
	for (int i = 0; i < iCycles; i++)
    {
        dProcessingTime += binaryDilationWrapper(d_img, imageW, imageH, threshold, iteration, brightness, contrast, mask_w, mask_h, 0);
    }

	// check if kernel execution generated an error and sync host
    cutilCheckMsg("Error: Kernel execution FAILED");
    // Get average computation time
    dProcessingTime /= (float)iCycles;

	printf("Processing Time of Binary Dilation Technique: %.5f\n\n", dProcessingTime);

	dProcessingTime = 0.0;
	for (int i = 0; i < iCycles; i++)
    {
        dProcessingTime += binarizationWrapper(d_img, imageW, imageH, threshold, brightness, contrast, 0);
    }

	// check if kernel execution generated an error and sync host
    cutilCheckMsg("Error: Kernel execution FAILED");
    // Get average computation time
    dProcessingTime /= (float)iCycles;

	printf("Processing Time of Binarization Technique: %.5f\n\n", dProcessingTime);

	dProcessingTime = 0.0;
	for (int i = 0; i < iCycles; i++)
    {
        dProcessingTime += grayErosionWrapper(d_img, imageW, imageH, threshold, iteration, brightness, contrast, mask_w, mask_h, 0);
    }

	// check if kernel execution generated an error and sync host
    cutilCheckMsg("Error: Kernel execution FAILED");
    // Get average computation time
    dProcessingTime /= (float)iCycles;

	printf("Processing Time of Gray Erosion Technique: %.5f\n\n", dProcessingTime);

	dProcessingTime = 0.0;
	for (int i = 0; i < iCycles; i++)
    {
        dProcessingTime += grayDilationWrapper(d_img, imageW, imageH, threshold, iteration, brightness, contrast, mask_w, mask_h, 0);
    }

	// check if kernel execution generated an error and sync host
    cutilCheckMsg("Error: Kernel execution FAILED");
    // Get average computation time
    dProcessingTime /= (float)iCycles;

	printf("Processing Time of Gray Dilation Technique: %.5f\n\n", dProcessingTime);

	dProcessingTime = 0.0;
	for (int i = 0; i < iCycles; i++)
    {
        dProcessingTime += runtime = highPassFilterWrapper(d_img, imageW, imageH, iteration, brightness, contrast, 0);
    }

	// check if kernel execution generated an error and sync host
    cutilCheckMsg("Error: Kernel execution FAILED");
    // Get average computation time
    dProcessingTime /= (float)iCycles;

	printf("Processing Time of High Pass Filtering Technique: %.5f\n\n", dProcessingTime);

	CUDA_UnbindTexture();
	cudaFree(d_img); 
	//throughput = (1.0e-6 * width * height)/dProcessingTime, dProcessingTime
}


void runBenchmarkCPU() 
{
	unsigned int *d_img;

	d_img = (unsigned int*)malloc(sizeof(unsigned int)*imageH*imageW);	
   
	// Start round-trip timer and process iCycles loops on the GPU
    const int iCycles = 10;

	printf("\nRunning a %d cycle benchmark on CPU...\n\n", iCycles);
	printf("Image Width: %d, Image Height: %d\n\n", imageW, imageH);


    double dProcessingTime = 0.0;
	for (int i = 0; i < iCycles; i++)
    {
        dProcessingTime += brightnessCPU(h_FSrc, d_img, imageW, imageH, brightness);
    }

    // check if kernel execution generated an error and sync host
    cutilCheckMsg("Error: Kernel execution FAILED");
    // Get average computation time
    dProcessingTime /= (double)iCycles;
	
	printf("Processing Time of Brightness Adjustment Technique: %.5f\n\n", dProcessingTime);
	
	dProcessingTime = 0.0;
	for (int i = 0; i < iCycles; i++)
    {
        dProcessingTime += gammaCorrectionCPU(h_FSrc, d_img, imageW, imageH, contrast);
    }

    // check if kernel execution generated an error and sync host
    cutilCheckMsg("Error: Kernel execution FAILED");
    // Get average computation time
    dProcessingTime /= (double)iCycles;
	
	printf("Processing Time of Gamma Correction Technique: %.5f\n\n", dProcessingTime);

		
	dProcessingTime = 0.0;
	for (int i = 0; i < iCycles; i++)
    {
        dProcessingTime += grayscaleCPU(h_FSrc, d_img, imageW, imageH, brightness, contrast, 0);
    }

    // check if kernel execution generated an error and sync host
    cutilCheckMsg("Error: Kernel execution FAILED");
    // Get average computation time
    dProcessingTime /= (double)iCycles;
	
	printf("Processing Time of Grayscale Technique: %.5f\n\n", dProcessingTime);
	
	dProcessingTime = 0.0;
	for (int i = 0; i < iCycles; i++)
    {
        dProcessingTime += invertCPU(h_FSrc, d_img, imageW, imageH, brightness, contrast, 0);
    }

	// check if kernel execution generated an error and sync host
    cutilCheckMsg("Error: Kernel execution FAILED");
    // Get average computation time
    dProcessingTime /= (double)iCycles;

	printf("Processing Time of Invert Technique: %.5f\n\n", dProcessingTime);
	
	dProcessingTime = 0.0;
	for (int i = 0; i < iCycles; i++)
    {
        dProcessingTime += edgeDetectCPU(h_Src, d_img, imageW, imageH, brightness, contrast, 0);
    }

	// check if kernel execution generated an error and sync host
    cutilCheckMsg("Error: Kernel execution FAILED");
    // Get average computation time
    dProcessingTime /= (double)iCycles;

	printf("Processing Time of Sobel Filter Technique: %.5f\n\n", dProcessingTime);

	dProcessingTime = 0.0;
	for (int i = 0; i < iCycles; i++)
    {
        dProcessingTime += smoothenCPU(h_FSrc, d_img, imageW, imageH, box_radius, iteration, brightness, contrast, 0);
    }

    // check if kernel execution generated an error and sync host
    cutilCheckMsg("Error: Kernel execution FAILED");
    // Get average computation time
    dProcessingTime /= (double)iCycles;
	
	printf("Processing Time of Mean Filtering Technique: %.5f\n\n", dProcessingTime);

	dProcessingTime = 0.0;
	for (int i = 0; i < iCycles; i++)
    {
        dProcessingTime += binarizeCPU(h_FSrc, d_img, imageW, imageH, threshold, brightness, contrast, 0);
    }

    // check if kernel execution generated an error and sync host
    cutilCheckMsg("Error: Kernel execution FAILED");
    // Get average computation time
    dProcessingTime /= (double)iCycles;
	
	printf("Processing Time of Binarization Technique: %.5f\n\n", dProcessingTime);
	
	dProcessingTime = 0.0;
	for (int i = 0; i < iCycles; i++)
    {
        dProcessingTime += binaryErosionCPU(h_FSrc, d_img, imageW, imageH, threshold, iteration, brightness, contrast, mask_w, mask_h, 0);
    }

	// check if kernel execution generated an error and sync host
    cutilCheckMsg("Error: Kernel execution FAILED");
    // Get average computation time
    dProcessingTime /= (double)iCycles;

	printf("Processing Time of Binary Erosion Technique: %.5f\n\n", dProcessingTime);

	dProcessingTime = 0.0;
	for (int i = 0; i < iCycles; i++)
    {
        dProcessingTime += binaryDilationCPU(h_FSrc, d_img, imageW, imageH, threshold, iteration, brightness, contrast, mask_w, mask_h, 0);
    }

	// check if kernel execution generated an error and sync host
    cutilCheckMsg("Error: Kernel execution FAILED");
    // Get average computation time
    dProcessingTime /= (double)iCycles;

	printf("Processing Time of Binary Dilation Technique: %.5f\n\n", dProcessingTime);

	dProcessingTime = 0.0;
	for (int i = 0; i < iCycles; i++)
    {
        dProcessingTime += grayErosionCPU(h_FSrc, d_img, imageW, imageH, threshold, iteration, brightness, contrast, mask_w, mask_h, 0);
    }

	// check if kernel execution generated an error and sync host
    cutilCheckMsg("Error: Kernel execution FAILED");
    // Get average computation time
    dProcessingTime /= (double)iCycles;

	printf("Processing Time of Gray Erosion Technique: %.5f\n\n", dProcessingTime);

	dProcessingTime = 0.0;
	for (int i = 0; i < iCycles; i++)
    {
        dProcessingTime += grayDilationCPU(h_FSrc, d_img, imageW, imageH, threshold, iteration, brightness, contrast, mask_w, mask_h, 0);
    }

	// check if kernel execution generated an error and sync host
    cutilCheckMsg("Error: Kernel execution FAILED");
    // Get average computation time
    dProcessingTime /= (double)iCycles;

	printf("Processing Time of Gray Dilation Technique: %.5f\n\n", dProcessingTime);
	
	dProcessingTime = 0.0;
	for (int i = 0; i < iCycles; i++)
    {
        dProcessingTime += sharpenCPU(h_FSrc, d_img, imageW, imageH, iteration, brightness, contrast, 0);	
    }

	// check if kernel execution generated an error and sync host
    cutilCheckMsg("Error: Kernel execution FAILED");
    // Get average computation time
    dProcessingTime /= (double)iCycles;

	printf("Processing Time of High Pass Filtering Technique: %.5f\n\n", dProcessingTime);
	
	free(d_img);

	//throughput = (1.0e-6 * width * height)/dProcessingTime
}



/*
	Main function
*/
int main(int argc, char **argv)
{
	// First load the image, so we know what the size of the image (imageW and imageH)    
	//image data is now in h_Src
	loadImageSDL(&h_Src, &imageW, &imageH, image_file); 	
	loadImageSDLFloat(&h_FSrc, &imageW, &imageH, image_file);	
	
	// First initialize OpenGL context, create opengl window
	initGL( &argc, argv );

	//get device with max gflops and use it as device for setting GL
	cudaGLSetGLDevice( cutGetMaxGflopsDeviceId() );
    cutilSafeCall( CUDA_MallocArray(&h_Src, imageW, imageH) );		

	int choice = 0;
	printf("Press [1] to run the main program\nPress [2] to run the benchmark on the GPU\nPress [3] to run benchmark on the CPU\nPress [4] to exit the program\nChoice: ");
	scanf("%d", &choice);

	if(choice == 1)
	{			
		initOpenGLBuffers();

		GLUI_Master.set_glutDisplayFunc(display);  
		GLUI_Master.set_glutReshapeFunc(reshape);  
		GLUI_Master.set_glutIdleFunc(idle);

		glutMainLoop();	

		cutilSafeCall(cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, gl_PBO, 
								   cudaGraphicsMapFlagsWriteDiscard));
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
		glDeleteBuffers(1, &gl_PBO);
		glDeleteTextures(1, &gl_Tex);

	}else if(choice == 2){

		CUDA_BindTexture();
		runBenchmarkGPU();
		getchar();

	}else if(choice == 3){

		CUDA_BindTexture();
		runBenchmarkCPU();
		getchar();

	}

	cutilSafeCall( CUDA_FreeArray() );
	cudaFree(d_temp);
	free(h_Src);		
	free(h_FSrc);		
	cutilExit(argc, argv);
	cudaThreadExit();
	exit(0);
	
}