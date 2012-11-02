CC = /opt/cuda/bin/nvcc 
# Remove -DGRAPHICS to compile with no graphic support (e.g. for measurements)
CFLAGS = -O2 
# -DGRAPHICS
RRDIR = ../lib

# Adapt this to your distribution if necessary
INCSDL =
LIBSDL = -L/usr/X11R6/lib64 


INC = $(INCSDL) -I$(RRDIR) 
LIB = $(LIBSDL) -L$(RRDIR) -lX11 -lgraphicsScreen -lSDL -lStopWatch -lpthread -lm

all: RoadMap Cuda

RoadMap: RoadMap.c
	$(CC) $(CFLAGS) RoadMap.c -o RoadMap $(INC) $(LIB)

Cuda: RoadMapCuda.cu
	$(CC) $(CFLAGS) RoadMapCuda.cu -o RoadMapCuda $(INC) $(LIB)
	
clean:
	rm -f RoadMap RoadMapCuda *~
