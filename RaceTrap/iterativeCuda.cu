/*
 * RaceTrap implementation based on RaceTrap.java
 *
 * Created on 22. juni 2000, 13:48
 * 
 * Brian Vinter
 * 
 * Modified by John Markus Bjørndalen, 2008-12-04, 2009-10-15. 
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include "graphicsScreen.h"
#include "StopWatch.h"
#include "stack.c"
#include <pthread.h>


typedef struct {
  int x; 
  int y; 
} Coord; 

typedef struct {
  double        length;     // Length of the current path (distance)
  unsigned char nCitiesVisited;    // Number of bags currently placed
  unsigned char path[0];    // Array of vertex/bag numbers in the path (see comment in
  // Alloc_RouteDefinition())
} RouteDefinition; 


pthread_mutex_t mut = PTHREAD_MUTEX_INITIALIZER;
RouteDefinition *bestRoute;

int      nTotalCities = 0;             // Number of grain-bagsg
Coord   *cityCoords;             // Coordinates for the grain-bags
double **distanceTable;         // Table of distances between any two grain-bags
double   maxRouteLen = 10E100;  // Initial best distance, must be longer than any possible route
double   globalBest  = 10E100;  // Bounding variable

int fanOutLevel = 3;


inline RouteDefinition* Alloc_RouteDefinition()
{
  if (nTotalCities <= 0) 
  {
    fprintf(stderr, "Error: Alloc_RouteDefinition called with invalid nTotalCities (%d)\n", nTotalCities); 
    exit(-1); 
  }
  // NB: The +nTotalCities*sizeof.. trick "expands" the path[0] array in RouteDefintion
  // to a path[nTotalCities] array.
  RouteDefinition *def = NULL;  
  return (RouteDefinition*) malloc(sizeof(RouteDefinition) + nTotalCities * sizeof(def->path[0]));
}



#ifdef GRAPHICS
// Plots a route on the display
void PlotRoute(char *path)
{ 
  int i;
  gs_clear(WHITE);
  // Plot each grain bag
  for(i = 0; i < nTotalCities; i++)
    gs_dot(cityCoords[i].x, cityCoords[i].y, 10, RED); 
  
  // Plot edges in the path
    for(i = 0; i < nTotalCities-1; i++)
      gs_line(cityCoords[(int)path[i  ]].x, cityCoords[(int)path[i  ]].y, 
	      cityCoords[(int)path[i+1]].x, cityCoords[(int)path[i+1]].y, RED);    
      
      // Plot the final edge closing the path
      gs_line(cityCoords[(int)path[nTotalCities-1]].x, cityCoords[(int)path[nTotalCities-1]].y, 
	      cityCoords[(int)path[0      ]].x, cityCoords[(int)path[0      ]].y, RED);
      
      gs_update();
}
#endif


int nodesAtLevel(int level)
{
  level = level - 1;
  int result = 1;
  for (int i = 1; i <= level; i++){
    result = result * (nTotalCities - i);
  }
  return result;
}

char* stackToArray(stack_t *stck)
{
  printf("Entering Flatten Stack\n");
  
  int elemSize = sizeof(RouteDefinition) + nTotalCities * sizeof(unsigned char);
  int rest = elemSize % 4;
  int padding = 0; 
  
  if (rest > 0){
    padding = 4 - rest;
  }
  
  int arraySize = (elemSize+padding) * nodesAtLevel(fanOutLevel);
  
  printf("Struct size is: %d, rest: %d, padding: %d, array: %d\n", elemSize, rest, padding, arraySize);

  char *tmp_route, *array, *iter;
  
  array = (char*)malloc(sizeof(unsigned char) * arraySize);
    
  iter = array;
  
  RouteDefinition *route;
  
  while(stck->size > 0){                
    
    tmp_route = (char*)pop_back(stck);
    
    memcpy(iter, tmp_route, elemSize);
        
    iter += (elemSize + padding);
    
    free(tmp_route);
  }
  
  return array;

}

/* 
 * A recursive Traveling Salesman Solver using branch-and-bound. 
 * 
 * Returns the shortest roundtrip path based on the starting path described in "route". 
 * 
 * The returned path is "route" if "route" already includes all the bags
 * in the route. If not, route will be freed, and a new path will be returned. 
 * 
 * NB: this function is destructive - it calls free() on route if it finds
 *     a better route than the provided route! 
 * 
 * NB2: there is a slight problem with the below code: ShortestRoute will return a 
 *      semi-intialized bestRoute if all the new permitations are longer than 
 *      globalBest. It shouldn't cause any problems though, as that route 
 *      will be thrown away. 
 */ 
RouteDefinition *ShortestRoute(RouteDefinition *route)
{ 
  printf("Size of route def: %d\n", sizeof(RouteDefinition));
  
  bestRoute = Alloc_RouteDefinition(); 
  
  bestRoute->length = 1.0;//maxRouteLen;
  
  int nodesAtThisLevel = nodesAtLevel(fanOutLevel); 
  //(nTotalCities - 1) * (nTotalCities - 2);
  printf("Nodes at this level: %d \n", nodesAtThisLevel);
  
  //stack_t* stck;
  //stck = stack_create();
  
  //    push(stck, (void*)&"a");
  //    push(stck, (void*)&"b");
  //    push(stck, (void*)&"c");
  //    push(stck, (void*)&"d");
  //    
  //    while(stck->size > 0)
  //    {
    //    	printf("Stack elem %s\n", pop(stck));
    //    	
    //    }
    //	
    //	
    //    
    //    // allocate an array of all routes at level one
    //    RouteDefinition *routes = (RouteDefinition*)malloc(sizeof(RouteDefinition) * nodesAtThisLevel);
    //    
    //        // Generate an array holding all routes at level two
    
    //	for(int p = route->nVisited; p < nTotalCities; p++){
      //		// Calculate route length
      //		double newLength = route->length + distanceTable[route->path[route->nCitiesVisited-1]][route->path[p]];  	
      //	
      //		// Copy current route to new route
      //		memcpy(routes[p-1].path, route->path, nTotalCities);			
      //		
      //	    // Swaps the position of bag # 'i' and bag # 'nCitiesVisited' from route
      //	    routes[p-1].path[route->nCitiesVisited] = route->path[p];
      //	    routes[p-1].path[p]              = route->path[route->nCitiesVisited]; 
      //	    routes[p-1].nCitiesVisited = route->nCitiesVisited + 1;
      //	    routes[p-1].length  = newLength;
      //	    
      //	    
      //	    // Print current calculatet paths
      //	    printf("Route: %d - path: ", (p-1));
      //	    for (int c = 0; c < nTotalCities; c++){
	//	    	printf("%d", routes[p-1].path[c]);
	//	    }
	//	    printf("\n");
	//    }
	
	// LEVEL 3 in three structure
	
	stack_t* stck;
	stck = stack_create();
	
	// This is the CUDA part of the assignment 
	//    for(int p = route->nCitiesVisited; p < nTotalCities; p++){
	  //        
	//        
	//        
	RouteDefinition *newRoute;
	//        newRoute = Alloc_RouteDefinition();  
	//        
	double newLength;// = route->length + distanceTable[route->path[route->nCitiesVisited-1]][route->path[p]];
	//        
	//        memcpy(newRoute->path, route->path, nTotalCities);   // Copy current route from route
	//            
	//        // Swaps the position of bag # 'i' and bag # 'nCitiesVisited' from route
	//        newRoute->path[route->nCitiesVisited] = route->path[p];
	//        newRoute->path[p]              = route->path[route->nCitiesVisited]; 
	//        newRoute->nCitiesVisited = route->nCitiesVisited + 1;
	//        newRoute->length  = newLength;
	//        
	push(stck, route);
	
	RouteDefinition *curr_route;
	curr_route = Alloc_RouteDefinition();
	
	while(stck->size > 0){
	  curr_route = (RouteDefinition *)pop_back(stck);
	  
	  // Has visited all cities
	  
	  if (curr_route->nCitiesVisited == fanOutLevel){
	    
	    // Push current popped node to stack
	    push(stck, curr_route);
	    
	    // convert stack to array
	    char *array = stackToArray(stck);
	    free(stck);

	    break;          
	  } 
	  
	  else {
	    for (int i = curr_route->nCitiesVisited; i < nTotalCities; i++){
	      
	      newLength = curr_route->length + distanceTable[curr_route->path[curr_route->nCitiesVisited-1]][curr_route->path[i]];
	      
	      newRoute = Alloc_RouteDefinition();  
	      
	      memcpy(newRoute->path, curr_route->path, nTotalCities);   // Copy current route from route
	      
	      // Swaps the position of bag # 'i' and bag # 'nCitiesVisited' from route
	      newRoute->path[curr_route->nCitiesVisited] = curr_route->path[i];
	      newRoute->path[i]              = curr_route->path[curr_route->nCitiesVisited]; 
	      newRoute->nCitiesVisited = curr_route->nCitiesVisited + 1;
	      newRoute->length  = newLength;
	      	      
	      push(stck, newRoute);
	      
	    }   
	  } 
	}   
	
	free(route);
	
	return bestRoute;
	}
	
	// In the desert, the shortest route is a straight line :)
	double EuclidDist(Coord *from, Coord *to)
	{ 
	  double dx = fabs(from->x - to->x);
	  double dy = fabs(from->y - to->y);
	  return sqrt(dx*dx + dy*dy);
	}
	
	// Reads coordinates from a file and generates a distance-table
	static void ReadRoute()
	{ 
	  FILE *file = fopen("./route.dat", "r");
	  int i,j;
	  
	  // Read how many bags there are
	  if (fscanf(file, "%d", &nTotalCities) != 1) 
	  {
	    printf("Error: couldn't read number of bags from route definition file.\n");
	    exit(-1);
	  }
	  
	  // Allocate array of bag coords. 
	  cityCoords = (Coord*) malloc(nTotalCities * sizeof(Coord)); 
	  
	  // Read the coordinates of each grain bag
	  for (i = 0; i < nTotalCities; i++)
	  {
	    if (fscanf(file,"%d %d", &cityCoords[i].x, &cityCoords[i].y) != 2) 
	    {
	      printf("Error: missing or invalid definition of coordinate %d.\n", i);
	      exit(-1);
	    }
	  }
	  
	  // Allocate distance table 
	  distanceTable = (double**) malloc(nTotalCities * sizeof(double*));
	  for (i = 0; i < nTotalCities; i++)
	    distanceTable[i] = (double*) malloc(nTotalCities * sizeof(double));
	  
	  // Compute the distances between each of the grain bags.
	    for (i = 0; i < nTotalCities; i++)	  
	      for (j = 0; j < nTotalCities; j++)	  
		distanceTable[i][j] = EuclidDist(&cityCoords[i], &cityCoords[j]);
	}
	
	
	int main (int argc, char **argv) 
	{
	  RouteDefinition *originalRoute, *res;
	  int i;
	  char buf[256];
	  
	  ReadRoute();
	  
	  #ifdef GRAPHICS
	  gs_init(501,501);
	  #endif
	  
	  // Set up an initial path that goes through each bag in turn. 
	  originalRoute = Alloc_RouteDefinition(); 
	  for (i = 0; i < nTotalCities; i++)
	    originalRoute->path[i] = (unsigned char) i;
	  
	  #ifdef GRAPHICS
	    // Show the original route
	    PlotRoute((char *)originalRoute->path); 
	    #endif
	    
	    originalRoute->length = 0.0;
	    originalRoute->nCitiesVisited = 1;
	    
	    sw_init();
	    sw_start();
	    // Find the best route
	    res = ShortestRoute(originalRoute);
	    
	    sw_stop();
	    sw_timeString(buf);
	    
	    printf("Route length is %lf it took %s\n", res->length, buf);
	      
	      #ifdef GRAPHICS
	      // Show the best route
	      PlotRoute((char *)res->path);
	      
	      free(res);
	      sleep(2);
	    gs_exit();
	    #endif  
	    
	    return 0;
	}
	