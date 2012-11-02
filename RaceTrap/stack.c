/* Author: Alexander Svendsen */
// #include "stack.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>


struct stack;
typedef struct stack stack_t;


struct stacknode;

typedef struct stacknode stacknode_t;

struct stacknode {
    stacknode_t *next;
    void *item;
};

struct stack {
    stacknode_t *head;
    stacknode_t *tail;
	int size;
};


void fatal_error()
{
    fprintf(stderr, "fatal error: %s\n", "out of memory");
    exit(1);
}

static stacknode_t *newnode(void *item)
{
    stacknode_t *node = (stacknode_t *)malloc(sizeof(stacknode_t));
    if (node == NULL)
	    fatal_error();
    node->next = NULL;
    node->item = item;
    return node;
}

/*
 * creates the stack
 * FILO = First Out, Last In
 */
stack_t *stack_create()
{
    stack_t *stack = (stack_t *)malloc(sizeof(stack_t));
    if (stack == NULL)
	    fatal_error();
    stack->head = NULL;
    stack->tail = NULL;
	stack->size = 0;
    return stack;
}

/*
 * Destroyes the stack and free all the items
 */
void stack_destroy(stack_t *stack)
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
void push(stack_t *stack, void *item)
{
    stacknode_t *node = newnode(item);
    if (stack->head == NULL) {
	    stack->head = stack->tail = node;
    }
    else {
	    node->next = stack->head;
	    stack->head = node;
    }
    stack->size++;

}


/*
 * Pop from the top of the stack
 */
void *pop(stack_t *stack)
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

