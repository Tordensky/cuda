#include "graphicsScreen.h"
#include "StopWatch.h"
#include <stdio.h>
#include <cuda.h>

#define WIDTH 5000
#define HEIGHT 5000
#define SIZE WIDTH*HEIGHT

int zooms=10;

int bytesize;
int *devResult, *result;

int threadsPerBlock;
int blocksPerGrid; 


#ifndef GRAPHICS
int crc;
#endif

int colortable[] = {
  WHITE, 
  WHITE, 
  BLUE, 
  CYAN, 
  BLACK, 
  GREEN, 
  MAGENTA, 
  ORANGE, 
  PINK, 
  RED,
  YELLOW
};

int colorcount=10;

float box_x_min, box_x_max, box_y_min, box_y_max;

__device__ inline float translate_x(int x) {       
  return (((box_x_max-box_x_min)/WIDTH)*x)+box_x_min;
}

__device__ inline float translate_y(int y) {
  return (((box_y_max-box_y_min)/HEIGHT)*y)+box_y_min;
}

__device__ inline int solve(float x, float y)              //Simple Mandelbrot
{                                          //divergation test
  float r=0.0,s=0.0;
  float next_r,next_s;
  int itt=0;
  
  while((r*r+s*s)<=4.0) {
    next_r=r*r-s*s+x;
    next_s=2*r*s+y;
    r=next_r; s=next_s;
    if(++itt==100)break;
  }
  
  return itt;
}

__global__ void cudaSolve(int *res, int width, int height, float x_min, float x_max, float y_min, float y_max)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    
    int n = (width*height);
    
    if (i < n){
      int x, y;
    
      // Calculate x and y from i
      x = int(i % width);
      y = int(i / width);
                  
      res[i] = solve((((x_max-x_min)/WIDTH)*x)+x_min, (((y_max-y_min)/HEIGHT)*y)+y_min) * 10 / 100; 
    }
    
    __syncthreads();
}


void CreateMap() {        //Our 'main' function
    
  
  
  // Solve on Cuda device
  cudaSolve<<<blocksPerGrid, threadsPerBlock>>>(devResult, WIDTH, HEIGHT, box_x_min, box_x_max, box_y_min, box_y_max);
  
  // Copy result back to host
  cudaMemcpy((void*)result, (void*)devResult, bytesize, cudaMemcpyDeviceToHost);
  
  
  
  int x, y;
  
  x = 0;
  y = 0;
  for (int i = 0; i < SIZE; i++){
    x = int(i % WIDTH);
    y = int(i / WIDTH);
            
    //printf("Kommer hit før segfault %d\n", i);
#ifdef GRAPHICS
    gs_plot(x, y, colortable[result[i]]); //Plot the coordinate to map
#else
    crc+=colortable[result[i]];
#endif
    }
#ifdef GRAPHICS
  gs_update();
#endif
  
   // TODO allocate and free before all and after all
  
  //result[0] = 123;
//   printf("New round\n");
//   for (int r = 0; r < SIZE; r++)
//     if (result[r] > 10)
//       printf("\tResult %d\n", result[r]);
  
//   for(y=0;y<HEIGHT;y++)              //Main loop for map generation
//     for(x=0;x<WIDTH;x++){  
//       color = solve(translate_x(x), translate_y(y))*colorcount/100;
//       
// #ifdef GRAPHICS
//       gs_plot(x,y,colortable[color]); //Plot the coordinate to map
// 
// #else
//       crc+=colortable[color];
// #endif
//     }
// #ifdef GRAPHICS
//   gs_update();
// #endif
}


int
RoadMap ()
{
    int i;
    double deltaxmin, deltaymin, deltaxmax,deltaymax;

    
    
  
    bytesize = SIZE * sizeof(int);
  
    // Allocate space for result on host
    printf("size is: %d, bytesize: %d\n", SIZE, bytesize);
  
    result = (int*)malloc(bytesize);
  
    // Allocate space for result on device 
    cudaMalloc((void**)&devResult, bytesize);

    // Calculate space and number of blocks
    threadsPerBlock = 512;
    blocksPerGrid = (SIZE + threadsPerBlock - 1) / threadsPerBlock;
  
    printf("Thread Per block: %d, blocksPerGrid: %d\n", threadsPerBlock, blocksPerGrid);

    box_x_min=-1.5; box_x_max=0.5;           //Set the map bounding box for total map
    box_y_min=-1.0; box_y_max=1.0;
    
    deltaxmin=(-0.9-box_x_min)/zooms;
    deltaxmax=(-0.65-box_x_max)/zooms;
    deltaymin=(-0.4-box_y_min)/zooms;
    deltaymax=(-0.1-box_y_max)/zooms;
    
    CreateMap();                     //Call our main
    for(i=0;i<zooms;i++){
      box_x_min+=deltaxmin;
      box_x_max+=deltaxmax;
      box_y_min+=deltaymin;
      box_y_max+=deltaymax;
      CreateMap();                     //Call our main
    }                       
    
    cudaFree(devResult);
  
    free(result);
    
    return 0;
}

int main (int argc, char *argv[]){
  char buf[256];

#ifdef GRAPHICS
  gs_init(WIDTH, HEIGHT);
#endif
  
  sw_init();
  sw_start();
  RoadMap();
  sw_stop();

  sw_timeString(buf);
  
  printf("Time taken: %s\n",buf);

#ifdef GRAPHICS
  gs_exit();
#else
  printf("CRC is %x\n",crc);
#endif

  return 0;
}
