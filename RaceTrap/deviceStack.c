/* Author: Alexander Svendsen */
// #include "stack.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>


// struct stack;
// typedef struct stack stack_t;
// 
// 
// struct stacknode;
// 
// typedef struct stacknode stacknode_t;
// 
// struct stacknode {
//     stacknode_t *next;
//     stacknode_t *prev;
//     void *item;
// };
// 
// struct stack {
//     stacknode_t *head;
//     stacknode_t *tail;
// 	int size;
// };


__device__ void device_fatal_error()
{
    printf("fatal error: %s\n", "out of memory");
    //exit(1);
}

__device__ static stacknode_t *deviceNewnode(void *item)
{
    stacknode_t *node = (stacknode_t *)malloc(sizeof(stacknode_t));
    if (node == NULL)
	    device_fatal_error();
    node->next = NULL;
    node->item = item;
    return node;
}

/*
 * creates the stack
 * FILO = First Out, Last In
 */
__device__ stack_t *device_stack_create()
{
    stack_t *stack = (stack_t *)malloc(sizeof(stack_t));
    if (stack == NULL)
	    device_fatal_error();
    stack->head = NULL;
    stack->tail = NULL;
	stack->size = 0;
    return stack;
}

/*
 * Destroyes the stack and free all the items
 */
__device__ void device_stack_destroy(stack_t *stack)
{
    stacknode_t *node = stack->head;
    while (node != NULL) {
	    stacknode_t *tmp = node;
	    node = node->next;
	    free(tmp);
    }
    free(stack);
}

/*
 * Pushes the item to the start of the stack
 */
__device__ void device_push(stack_t *stack, void *item)
{
    stacknode_t *node = deviceNewnode(item);
    if (stack->head == NULL) {
	    stack->head = stack->tail = node;
    }
    else {
	    node->next = stack->head;
	    stack->head = node;
    }
    stack->size++;

}

__device__ void *device_pop_back(stack_t * stack)
{
	if (stack->head == NULL){
		return NULL;
	} else {
		void *item = stack->tail->item;
		stacknode_t *tmp = stack->tail;
		stack->tail = tmp->prev;
		
		if (stack->tail == NULL){
			stack->head = NULL;
		}
		
		free(tmp);
		stack->size--;
		return item;
	}
}

/*
 * Pop from the top of the stack
 */
__device__ void *device_pop(stack_t *stack)
{
	if (stack->head == NULL) {
		return NULL;
    }
    else {
        void *item = stack->head->item;
	    stacknode_t *tmp = stack->head;
	    stack->head = stack->head->next;
	    if (stack->head == NULL) {
	        stack->tail = NULL;
	    }
	    free(tmp);
		stack->size--;
	    return item;
	
    }
    
	
}

