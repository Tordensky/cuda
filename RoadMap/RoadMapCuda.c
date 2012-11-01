#include "graphicsScreen.h"
#include "StopWatch.h"
#include <stdio.h>

#define WIDTH 500
#define HEIGHT 500
#define SIZE WIDTH*HEIGHT

int zooms=10;

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

double box_x_min, box_x_max, box_y_min, box_y_max;

inline double translate_x(int x) {       
  return (((box_x_max-box_x_min)/WIDTH)*x)+box_x_min;
}

inline double translate_y(int y) {
  return (((box_y_max-box_y_min)/HEIGHT)*y)+box_y_min;
}

inline int solve(double x, double y)              //Simple Mandelbrot
{                                          //divergation test
  double r=0.0,s=0.0;
  double next_r,next_s;
  int itt=0;
  
  while((r*r+s*s)<=4.0) {
    next_r=r*r-s*s+x;
    next_s=2*r*s+y;
    r=next_r; s=next_s;
    if(++itt==100)break;
  }
  
  return itt;
}
  
void CreateMap() {        //Our 'main' function
  
  int x,y,color;                               //Holds the result of slove
  

  for(y=0;y<HEIGHT;y++)              //Main loop for map generation
    for(x=0;x<WIDTH;x++){  
      color=solve(translate_x(x),translate_y(y))*colorcount/100;
#ifdef GRAPHICS
      gs_plot(x,y,colortable[color]); //Plot the coordinate to map
#else
      crc+=colortable[color];
#endif
    }
#ifdef GRAPHICS
  gs_update();
#endif
}


int
RoadMap ()
{
  int i;
  double deltaxmin, deltaymin, deltaxmax,deltaymax;

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
