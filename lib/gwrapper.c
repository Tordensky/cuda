#include "graphicsScreen.h"

void gs_init_(int *xsize, int *ysize){gs_init(*xsize, *ysize);}
void gs_update_(){gs_update();}
void gs_exit_(){ gs_exit();}
void gs_plot_(int *x, int *y, int *color){gs_plot(*x, *y, *color);}
//int gs_plot_(int *x, int *y, int *color){printf("%d %d %x\n",*x, *y, *color); return 1;}

