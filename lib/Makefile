CC = gcc -O2

all: libgraphicsscreen.a libStopWatch.a

libgraphicsscreen.a: graphicsScreen.o
	ar rsv libgraphicsScreen.a graphicsScreen.o

graphicsScreen.o: graphicsScreen.c
	$(CC) $(INC) -c graphicsScreen.c

libStopWatch.a: StopWatch.o
	ar rsv libStopWatch.a StopWatch.o

StopWatch.o: StopWatch.c
	$(CC) $(INC) -c StopWatch.c

clean:
	rm -f *.o *~ *.a
