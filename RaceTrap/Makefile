CC = /opt/cuda/bin/nvcc 
# Remove -DGRAPHICS to compile with no graphic support (e.g. for measurements)
CFLAGS = -arch=sm_20
#-DGRAPHICS -O2
RRDIR = ../lib

# Adapt this to your distribution if necessary
INCSDL = 
LIBSDL = -L/usr/X11R6/lib64 


INC = $(INCSDL) -I$(RRDIR) 
LIB = $(LIBSDL) -L$(RRDIR) -lX11 -lgraphicsScreen -lSDL -lStopWatch -lpthread -lm

all: RaceTrap Cuda

RaceTrap: RaceTrap.c
	$(CC) $(CFLAGS) RaceTrap.c -o RaceTrap $(INC) $(LIB)

Cuda: iterativeCuda.cu
	$(CC) $(CFLAGS) iterativeCuda.cu -o RaceTrapCuda $(INC) $(LIB)
	
clean:
	rm -f RaceTrap RaceTrapCuda *~

