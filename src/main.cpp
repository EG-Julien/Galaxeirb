#include <stdio.h>
#include <stdbool.h>
#include <math.h>
#include <sys/time.h>
//#include <omp.h>



#include <string.h>
#include <time.h>

#include "cuda_runtime.h"
#include "kernel.cuh"

#include "GL/glew.h"

#include "SDL2/SDL.h"
#include "SDL2/SDL_opengl.h"

#include "text.h"


static int M = 10;
static float g_inertia = 0.5f;

static float oldCamPos[] = { 0.0f, 0.0f, -45.0f };
static float oldCamRot[] = { 0.0f, 0.0f, 0.0f };
static float newCamPos[] = { 0.0f, 0.0f, -45.0f };
static float newCamRot[] = { 0.0f, 0.0f, 0.0f };

static bool g_showGrid = true;
static bool g_showAxes = true;

float temps = 0.1;
int yolo = 0;



inline bool CUDA_MALLOC( void ** devPtr, size_t size ) {
	cudaError_t cudaStatus;
	cudaStatus = cudaMalloc( devPtr, size );
	if ( cudaStatus != cudaSuccess ) {
		printf( "error: unable to allocate buffer\n");
		return false;
	}
	return true;
}

inline bool CUDA_MEMCPY( void * dst, const void * src, size_t count, enum cudaMemcpyKind kind ) {
	cudaError_t cudaStatus;
	cudaStatus = cudaMemcpy( dst, src, count, kind );
	if ( cudaStatus != cudaSuccess ) {
		printf( "error: unable to copy buffer\n");
		return false;
	}
	return true;
}
	
void RandomizeFloatArray( int n, float * arr ) {
	for ( int i = 0; i < n; i++ ) {
		arr[i] = (float)rand() / ( (float)RAND_MAX / 2.0f ) - 1.0f;
	}
}

//------------------------------//


// Grid display
void DrawGridXZ( float ox, float oy, float oz, int w, int h, float sz ) {
	
	int i;

	glLineWidth( 1.0f );

	glBegin( GL_LINES );

	glColor3f( 0.48f, 0.48f, 0.48f );

	for ( i = 0; i <= h; ++i ) {
		glVertex3f( ox, oy, oz + i * sz );
		glVertex3f( ox + w * sz, oy, oz + i * sz );
	}

	for ( i = 0; i <= h; ++i ) {
		glVertex3f( ox + i * sz, oy, oz );
		glVertex3f( ox + i * sz, oy, oz + h * sz );
	}

	glEnd();

}


// Axis display
void ShowAxes() {

	glLineWidth( 2.0f );

	glBegin( GL_LINES );
	
	glColor3f( 1.0f, 0.0f, 0.0f );
	glVertex3f( 0.0f, 0.0f, 0.0f );
	glVertex3f( 2.0f, 0.0f, 0.0f );

	glColor3f( 0.0f, 1.0f, 0.0f );
	glVertex3f( 0.0f, 0.0f, 0.0f );
	glVertex3f( 0.0f, 2.0f, 0.0f );

	glColor3f( 0.0f, 0.0f, 1.0f );
	glVertex3f( 0.0f, 0.0f, 0.0f );
	glVertex3f( 0.0f, 0.0f, 2.0f );
	
	glEnd();

}

//Points display - unused
void showPoint() {
	int i = 0;
	float e = 0.0f;


	for(i=0; i<100; i++){
		glPointSize(10.0f);
		glColor3f(0.0f, 1.0f, 0.0f);
		glBegin( GL_POINTS );
		glVertex3f(0.0f, e, 0.0f);	
		glEnd();
		e++;
	}		
}

//----------------------------//

// Gather data from text file
void GetDataGalaxy(int numPoints, int tmp, float Masse[], float PosX[], float PosY[], float PosZ[], float VelX[], float VelY[], float VelZ[]) {

	FILE* f = NULL;
	f = fopen("dubinski.tab", "r");
	int error = 0;
	float Masst, PosXt, PosYt, PosZt, VelXt, VelYt, VelZt;
	int i;

	for(i = 0; i < numPoints; i++) {
		if(f != NULL) {
			fscanf(f, "%f %f %f %f %f %f %f\n", &Masst, &PosXt, &PosYt, &PosZt, &VelXt, &VelYt, &VelZt);
		
			Masse[i] = Masst;
			PosX[i] = PosXt;
			PosY[i] = PosYt;
			PosZ[i] = PosZt;
			VelX[i] = VelXt;
			VelY[i] = VelYt;
			VelZ[i] = VelZt;	
	
			for(int j = 0; j < tmp - 1; j++){
				fscanf(f, "%f %f %f %f %f %f %f\n", &Masst, &PosXt, &PosYt, &PosZt, &VelXt, &VelYt, &VelZt);
				//printf("bonjour\n");
			}

		} else {
			printf( "error\n" );
			error = 1;
		}
		//printf("%f, %f\n", Mass[i], PosX[i]);
	}
	fclose( f );
	printf("done\n");
}

//Galaxy display
void PrintGalaxy(int numPoints, int numThreads, int tmp, float Masse[], float PosX[], float PosY[], float PosZ[], float VelX[], float VelY[], float VelZ[], float accX[], float accY[], float accZ[]){
	int i;
	
	float * deviceSrc1 = NULL;
	float * deviceSrc2 = NULL;
	float * deviceSrc3 = NULL;
	float * deviceSrc4 = NULL;
	float * deviceSrc5 = NULL;
	float * deviceSrc6 = NULL;
	float * deviceSrc7 = NULL;
	float * deviceSrc8 = NULL;
	float * deviceSrc9 = NULL;
	float * deviceSrc10 = NULL;

	srand( (unsigned int)time( NULL ) );

	cudaError_t cudaStatus;

 	if (yolo == 0){
		cudaStatus = cudaDeviceReset();
		yolo = 1;
	}

	cudaStatus = cudaSetDevice( 0 );

	if ( cudaStatus != cudaSuccess ) {
		printf( "error: unable to setup cuda device\n");
	}

	CUDA_MALLOC( (void**)&deviceSrc1, numPoints * sizeof( float ) );
	CUDA_MALLOC( (void**)&deviceSrc2, numPoints * sizeof( float ) );
	CUDA_MALLOC( (void**)&deviceSrc3, numPoints * sizeof( float ) );
	CUDA_MALLOC( (void**)&deviceSrc4, numPoints * sizeof( float ) );
	CUDA_MALLOC( (void**)&deviceSrc5, numPoints * sizeof( float ) );
	CUDA_MALLOC( (void**)&deviceSrc6, numPoints * sizeof( float ) );
	CUDA_MALLOC( (void**)&deviceSrc7, numPoints * sizeof( float ) );
	CUDA_MALLOC( (void**)&deviceSrc8, numPoints * sizeof( float ) );
	CUDA_MALLOC( (void**)&deviceSrc9, numPoints * sizeof( float ) );
	CUDA_MALLOC( (void**)&deviceSrc10, numPoints * sizeof( float ) );


	CUDA_MEMCPY( deviceSrc1, Masse, numPoints * sizeof( float ), cudaMemcpyHostToDevice );
	CUDA_MEMCPY( deviceSrc2, PosX, numPoints * sizeof( float ), cudaMemcpyHostToDevice );
	CUDA_MEMCPY( deviceSrc3, PosY, numPoints * sizeof( float ), cudaMemcpyHostToDevice );
	CUDA_MEMCPY( deviceSrc4, PosZ, numPoints * sizeof( float ), cudaMemcpyHostToDevice );
	CUDA_MEMCPY( deviceSrc5, VelX, numPoints * sizeof( float ), cudaMemcpyHostToDevice );
	CUDA_MEMCPY( deviceSrc6, VelY, numPoints * sizeof( float ), cudaMemcpyHostToDevice );
	CUDA_MEMCPY( deviceSrc7, VelZ, numPoints * sizeof( float ), cudaMemcpyHostToDevice );
	CUDA_MEMCPY( deviceSrc8, accX, numPoints * sizeof( float ), cudaMemcpyHostToDevice );
	CUDA_MEMCPY( deviceSrc9, accY, numPoints * sizeof( float ), cudaMemcpyHostToDevice );
	CUDA_MEMCPY( deviceSrc10, accZ, numPoints * sizeof( float ), cudaMemcpyHostToDevice );


	int numBlocks = ( numPoints + ( numThreads - 1 ) ) / numThreads;

	saxpy( numBlocks, numThreads, numPoints, deviceSrc1, deviceSrc2, deviceSrc3, deviceSrc4, deviceSrc5, deviceSrc6, deviceSrc7, deviceSrc8, deviceSrc9, deviceSrc10 );

	cudaStatus = cudaDeviceSynchronize();

	if ( cudaStatus != cudaSuccess ) {
		printf( "error: unable to synchronize threads\n");
	}

	CUDA_MEMCPY( Masse, deviceSrc1, numPoints * sizeof( float ), cudaMemcpyDeviceToHost );
	CUDA_MEMCPY( PosX, deviceSrc2, numPoints * sizeof( float ), cudaMemcpyDeviceToHost );
	CUDA_MEMCPY( PosY, deviceSrc3, numPoints * sizeof( float ), cudaMemcpyDeviceToHost );
	CUDA_MEMCPY( PosZ, deviceSrc4, numPoints * sizeof( float ), cudaMemcpyDeviceToHost );
	CUDA_MEMCPY( VelX, deviceSrc5, numPoints * sizeof( float ), cudaMemcpyDeviceToHost );
	CUDA_MEMCPY( VelY, deviceSrc6, numPoints * sizeof( float ), cudaMemcpyDeviceToHost );
	CUDA_MEMCPY( VelZ, deviceSrc7, numPoints * sizeof( float ), cudaMemcpyDeviceToHost );
	CUDA_MEMCPY( accX, deviceSrc8, numPoints * sizeof( float ), cudaMemcpyDeviceToHost );
	CUDA_MEMCPY( accY, deviceSrc9, numPoints * sizeof( float ), cudaMemcpyDeviceToHost );
	CUDA_MEMCPY( accZ, deviceSrc10, numPoints * sizeof( float ), cudaMemcpyDeviceToHost );

	if (cudaStatus != cudaSuccess) {
		printf( "(EE) Unable to reset device\n" );
	}

//-----Fin cuda------//

	glPointSize(1.0f);

	glBegin( GL_POINTS );

	//#pragma omp parallel for
	for(i = 0; i < numPoints - 1; i++){
			
		VelX[i] += accX[i]; 
		VelY[i] += accY[i]; 
		VelZ[i] += accZ[i]; 
		PosX[i] += VelX[i]*temps; 
		PosY[i] += VelY[i]*temps; 
		PosZ[i] += VelZ[i]*temps; 
	
		if (i<= 16384 / tmp) 
			glColor3f(1.0f, 0.0f, 1.0f);
		else if (i > 32768 / tmp && i <= 40960 / tmp) 
			glColor3f(1.0f, 0.0f, 1.0f);
		else if (i > 49152 / tmp && i <= 65536 / tmp) 
			glColor3f(1.0f, 0.0f, 1.0f);
		else {
			glColor3f(0.0f, 1.0f, 1.0f);
		}
		glVertex3f(PosX[i], PosY[i], PosZ[i]);	
	}
	glEnd();	
}

int main( int argc, char ** argv ) {
	
	if (argc != 3) {
		printf( "usage: opengl numPoints numThreads\n" );
		return 0;
	}

	int numPoints = atoi( argv[1] );
	int numThreads = atoi( argv[2] );
	int tmp = 81920 / numPoints;

	float Masse[numPoints];
	float PosX[numPoints];
	float PosY[numPoints];
	float PosZ[numPoints];
	float VelX[numPoints];
	float VelY[numPoints];
	float VelZ[numPoints];
	float accX[numPoints];
	float accY[numPoints];
	float accZ[numPoints];

	SDL_Event event;
	SDL_Window * window;
	SDL_DisplayMode current;
	//omp_set_num_threads(4);
  	
	int width = 640;
	int height = 480;

	bool done = false;

	float mouseOriginX = 0.0f;
	float mouseOriginY = 0.0f;

	float mouseMoveX = 0.0f;
	float mouseMoveY = 0.0f;

	float mouseDeltaX = 0.0f;
	float mouseDeltaY = 0.0f;

	

	struct timeval begin, end;
	float fps = 0.0;
	char sfps[40] = "FPS: ";

	if ( SDL_Init ( SDL_INIT_EVERYTHING ) < 0 ) {
		printf( "error: unable to init sdl\n" );
		return -1;
	}

	if ( SDL_GetDesktopDisplayMode( 0, &current ) ) {
		printf( "error: unable to get current display mode\n" );
		return -1;
	}

	window = SDL_CreateWindow( "SDL", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, width, height, SDL_WINDOW_OPENGL );
  
	SDL_GLContext glWindow = SDL_GL_CreateContext( window );

	GLenum status = glewInit();

	if ( status != GLEW_OK ) {
		printf( "error: unable to init glew\n" );
		return -1;
	}

	if ( ! InitTextRes( "./bin/DroidSans.ttf" ) ) {
		printf( "error: unable to init text resources\n" );
		return -1;
	}

	SDL_GL_SetSwapInterval( 1 );
	GetDataGalaxy(numPoints, tmp, Masse, PosX, PosY, PosZ, VelX, VelY, VelZ);
	 
	while ( !done ) {
  		
		int i;

		while ( SDL_PollEvent( &event ) ) {
      
			unsigned int e = event.type;
			
			if ( e == SDL_MOUSEMOTION ) {
				mouseMoveX = event.motion.x;
				mouseMoveY = height - event.motion.y - 1;
			} else if ( e == SDL_KEYDOWN ) {
				if ( event.key.keysym.sym == SDLK_F1 ) {
					g_showGrid = !g_showGrid;
				} else if ( event.key.keysym.sym == SDLK_F2 ) {
					g_showAxes = !g_showAxes;
				} else if ( event.key.keysym.sym == SDLK_ESCAPE ) {
 					done = true;
				}
			}

			if ( e == SDL_QUIT ) {
				printf( "quit\n" );
				done = true;
			}
		}

		mouseDeltaX = mouseMoveX - mouseOriginX;
		mouseDeltaY = mouseMoveY - mouseOriginY;

		if ( SDL_GetMouseState( 0, 0 ) & SDL_BUTTON_LMASK ) {
			oldCamRot[ 0 ] += -mouseDeltaY / 5.0f;
			oldCamRot[ 1 ] += mouseDeltaX / 5.0f;
		}else if ( SDL_GetMouseState( 0, 0 ) & SDL_BUTTON_RMASK ) {
			oldCamPos[ 2 ] += ( mouseDeltaY / 100.0f ) * 0.5 * fabs( oldCamPos[ 2 ] );
			oldCamPos[ 2 ]  = oldCamPos[ 2 ] > -5.0f ? -5.0f : oldCamPos[ 2 ];
		}

		mouseOriginX = mouseMoveX;
		mouseOriginY = mouseMoveY;

		glViewport( 0, 0, width, height );
		glClearColor( 0.2f, 0.2f, 0.2f, 1.0f );
		glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
		glEnable( GL_BLEND );
		glBlendEquation( GL_FUNC_ADD );
		glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA );
		glDisable( GL_TEXTURE_2D );
		glEnable( GL_DEPTH_TEST );
		glMatrixMode( GL_PROJECTION );
		glLoadIdentity();
		gluPerspective( 50.0f, (float)width / (float)height, 0.1f, 100000.0f );
		glMatrixMode( GL_MODELVIEW );
		glLoadIdentity();

		for ( i = 0; i < 3; ++i ) {
			newCamPos[ i ] += ( oldCamPos[ i ] - newCamPos[ i ] ) * g_inertia;
			newCamRot[ i ] += ( oldCamRot[ i ] - newCamRot[ i ] ) * g_inertia;
		}

		glTranslatef( newCamPos[0], newCamPos[1], newCamPos[2] );
		glRotatef( newCamRot[0], 1.0f, 0.0f, 0.0f );
		glRotatef( newCamRot[1], 0.0f, 1.0f, 0.0f );
		
		if ( g_showGrid ) {
			DrawGridXZ( -100.0f, 0.0f, -100.0f, 20, 20, 10.0 );
		}

		if ( g_showAxes ) {
			ShowAxes();
		}

		gettimeofday( &begin, NULL );

		// Simulation should be computed here

		PrintGalaxy(numPoints, numThreads, tmp, Masse, PosX, PosY, PosZ, VelX, VelY, VelZ, accX, accY, accZ);

		//end of simulation code 

		gettimeofday( &end, NULL );

		fps = (float)1.0f / ( ( end.tv_sec - begin.tv_sec ) * 1000000.0f + end.tv_usec - begin.tv_usec) * 1000000.0f;
		sprintf( sfps, "FPS : %.4f", fps );

		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();
		gluOrtho2D(0, width, 0, height);
		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();

		DrawText( 10, height - 20, sfps, TEXT_ALIGN_LEFT, RGBA(255, 255, 255, 255) );
		DrawText( 10, 30, "'F1' : show/hide grid", TEXT_ALIGN_LEFT, RGBA(255, 255, 255, 255) );
		DrawText( 10, 10, "'F2' : show/hide axes", TEXT_ALIGN_LEFT, RGBA(255, 255, 255, 255) );

		SDL_GL_SwapWindow( window );
		SDL_UpdateWindowSurface( window );
	}

	SDL_GL_DeleteContext( glWindow );
	DestroyTextRes();
	SDL_DestroyWindow( window );
	SDL_Quit();

	return 1;
}

