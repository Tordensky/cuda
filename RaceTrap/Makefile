CC = /opt/cuda/bin/nvcc

# Remove -DGRAPHICS to compile with no graphic support (e.g. for measurements)
CFLAGS = -O2 -DGRAPHICS

# Use this for SDL
RRLIB = ../lib
INC = -I$(RRLIB) `sdl-config --cflags`
LIB = -L$(RRLIB) -L/usr/X11R6/lib64 -L/usr/X11R6/lib -lX11 -lgraphicsScreen -lStopWatch  `sdl-config --libs`

all: RaceTrap

RaceTrap: RaceTrap.c
	$(CC) $(CFLAGS) RaceTrap.c -o RaceTrap $(INC) $(LIB)

clean:
	rm -f *~ *.o core* RaceTrap

