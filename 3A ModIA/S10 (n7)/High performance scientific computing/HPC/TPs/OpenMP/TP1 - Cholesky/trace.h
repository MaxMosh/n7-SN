#include <string.h>
#include <stdio.h>


#define MAXEVENTS 30000
#define MAXTHREADS 100

long usecs ();


typedef struct event_struct{
  int type;
  long t_start, t_stop;
} Event;


void trace_init();
void trace_event_start(int type);
void trace_event_stop(int type);
void trace_dump(char *);

