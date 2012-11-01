#include <stdlib.h>
#include <math.h>
#include <SDL/SDL.h>
#include <signal.h>

SDL_Surface *gs_screen;
int gs_xsize, gs_ysize;


void gs_update_rect (int x, int y, int width, int height)
{
  SDL_UpdateRect(gs_screen, x, y, width, height);
}

void gs_update ()
{
  gs_update_rect (0, 0, gs_xsize, gs_ysize);
}

inline void __gs_plot (int x, int y, int gs_color)
{
  int R = gs_color>>16 & 0xFF;
  int G = gs_color>>8 & 0xFF;
  int B = gs_color & 0xFF;
  Uint32 color = SDL_MapRGB(gs_screen->format, R, G, B);

  switch (gs_screen->format->BytesPerPixel) {
  case 1: { /* Assuming 8-bpp */
    Uint8 *bufp;
    
    bufp = (Uint8 *)gs_screen->pixels + y*gs_screen->pitch + x;
    *bufp = color;
  }
    break;
    
  case 2: { /* Probably 15-bpp or 16-bpp */
    Uint16 *bufp;
    
    bufp = (Uint16 *)gs_screen->pixels + y*gs_screen->pitch/2 + x;
    *bufp = color;
  }
    break;
    
  case 3: { /* Slow 24-bpp mode, usually not used */
    Uint8 *bufp;
    
    bufp = (Uint8 *)gs_screen->pixels + y*gs_screen->pitch + x;
    *(bufp+gs_screen->format->Rshift/8) = R;
    *(bufp+gs_screen->format->Gshift/8) = G;
    *(bufp+gs_screen->format->Bshift/8) = B;
  }
    break;
    
  case 4: { /* Probably 32-bpp */
    Uint32 *bufp;
    
    bufp = (Uint32 *)gs_screen->pixels + y*gs_screen->pitch/4 + x;
    *bufp = color;
  }
    break;
  } 
}

void gs_plot (int x, int y, int color)
{
  if (SDL_MUSTLOCK(gs_screen)) {
    if (SDL_LockSurface(gs_screen) < 0) {
      return;
    }
  }

  __gs_plot (x, y, color);

  if ( SDL_MUSTLOCK(gs_screen) ) {
    SDL_UnlockSurface(gs_screen);
  }
}

static void signal_handler (int signal)
{
  switch (signal) {
  case SIGINT:
  case SIGTERM:
    exit (0);
  }
}



void gs_init(int xsize, int ysize)
{
  gs_xsize=xsize;
  gs_ysize=ysize;
  
  if (SDL_Init (SDL_INIT_VIDEO) < 0 ) {
    fprintf(stderr, "Unable to init SDL: %s\n", SDL_GetError());
    exit(1);
  }
  atexit(SDL_Quit); /* Just in case the user doesn't call gs_exit */
  //signal (SIGINT, signal_handler);
  //signal (SIGTERM, signal_handler);

  gs_screen = SDL_SetVideoMode (xsize, ysize, 16, SDL_SWSURFACE);
  if (gs_screen == NULL) {
    fprintf(stderr, "Unable to open window: %s\n", SDL_GetError());
    exit(1);
  }
  
  SDL_WM_SetCaption("GraphicsScreen", NULL);
}

void gs_init_with_title(int xsize, int ysize, const char* title)
{
  gs_init (xsize, ysize);
  SDL_WM_SetCaption(title, NULL);  
}


void gs_exit()
{
  SDL_Quit ();
}

void gs_clear (int gs_color)
{
  int R = gs_color>>16 & 0xFF;
  int G = gs_color>>8 & 0xFF;
  int B = gs_color & 0xFF;
  Uint32 color = SDL_MapRGB(gs_screen->format, R, G, B);
  SDL_Rect rect = {0, 0, gs_xsize, gs_ysize};
  
  SDL_FillRect (gs_screen, &rect, color);
}

void gs_dot (int x, int y, int z, int color)
{
  int i,j;

  if (SDL_MUSTLOCK(gs_screen)) {
    if (SDL_LockSurface(gs_screen) < 0) {
      return;
    }
  }

  for (j=y-z/2; j<y+z/2; j++)
    for(i=x-z/2; i<x+z/2; i++)      
      if(pow(((double)x-i)/((double)z/2.0),2.0)+pow(((double)y-j)/((double)z/2.0),2.0)<1)
	__gs_plot(i,j, color);

  if ( SDL_MUSTLOCK(gs_screen) ) {
    SDL_UnlockSurface(gs_screen);
  }
}

void gs_line (int x0, int y0, int x1, int y1, int color)
{
  int dy = y1 - y0;
  int dx = x1 - x0;
  float t = (float) 0.5;                      // offset for rounding
  float m;

  if (SDL_MUSTLOCK(gs_screen)) {
    if (SDL_LockSurface(gs_screen) < 0) {
      return;
    }
  }

  __gs_plot(x0, y0, color);
  if (fabs(dx) > fabs(dy)) {          // slope < 1
    m = (float) dy / (float) dx;      // compute slope
    t += y0;
    dx = (dx < 0) ? -1 : 1;
    m *= dx;
    while (x0 != x1) {
      x0 += dx;                           // step to next x value
      t += m;                             // add slope to y value
      __gs_plot(x0, (int) t, color);
    }
  } else {                                    // slope >= 1
    m = (float) dx / (float) dy;      // compute slope
    t += x0;
    dy = (dy < 0) ? -1 : 1;
    m *= dy;
    while (y0 != y1) {
      y0 += dy;                           // step to next y value
      t += m;                             // add slope to x value
      __gs_plot((int) t, y0, color);
    }
  }

  if ( SDL_MUSTLOCK(gs_screen) ) {
    SDL_UnlockSurface(gs_screen);
  }
}
