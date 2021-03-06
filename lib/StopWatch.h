#ifndef _STOPWATCH_H
#define	_STOPWATCH_H

#ifdef	__cplusplus
extern "C" {
#endif

#include <sys/time.h>
#include <string.h>
#include <stdio.h>

void sw_init();

void sw_start();

void sw_stop();

int readDays();

int readHours();

int readMinutes();

int readSeconds();

int readmSeconds();

void sw_timeString(char *buf);


#ifdef	__cplusplus
}
#endif

#endif	/* _STOPWATCH_H */
