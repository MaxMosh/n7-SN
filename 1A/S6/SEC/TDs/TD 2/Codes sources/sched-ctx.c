#define _XOPEN_SOURCE 600

#include <stdio.h>
#include <setjmp.h>
#include <signal.h>
#include <stdlib.h>
#include <unistd.h>
#include <ucontext.h>

#define MAX_THREAD 3
#define STACK_SIZE 16384
#define TIME_SLICE 4

void thread0();
void thread1();
void thread2();
void schedule(int sig);

ucontext_t uctx_main;

void (*thread_routine[MAX_THREAD])() = {thread0, thread1, thread2};
ucontext_t thread_save[MAX_THREAD];
char thread_stack[MAX_THREAD][STACK_SIZE];
int thread_state[MAX_THREAD];
int current;

void thread0() {
	int i,k;
	for (i=0;i<10;i++) {
		printf("thread 0\n");
		sleep(1);
	}
}
void thread1() {
	int i,k;
	for (i=0;i<10;i++) {
		printf("thread 1\n");
		sleep(1);
	}
}
void thread2() {
	int i,k;
	for (i=0;i<10;i++) {
		printf("thread 2\n");
		sleep(1);
	}
}


void schedule(int sig) {
	int k, old;
	alarm(TIME_SLICE);
	old = current;
	for (k=0;k<MAX_THREAD;k++) {
		current = (current + 1) % MAX_THREAD;
		if (thread_state[current] == 1) break;
	}
	if (k==MAX_THREAD) {
		printf("last thread completed: exiting\n");
		exit(0);
	}
	printf("schedule: save(%d) restore (%d)\n",old, current);
	if (swapcontext(&thread_save[old], &thread_save[current]) == -1)
		{ perror("swapcontext"); exit(0); }
}


int main() {
	int i;
	for (i=0;i<MAX_THREAD;i++) {
		if (getcontext(&thread_save[i]) == -1) 
			{ perror("getcontext"); exit(0); }
           	thread_save[i].uc_stack.ss_sp = thread_stack[i];
		thread_save[i].uc_stack.ss_size = sizeof(thread_stack[i]);
           	thread_save[i].uc_link = &uctx_main;
           	makecontext(&thread_save[i], thread_routine[i], 0);
		thread_state[i] = 1;
		printf("main: thread %d created\n",i);
	}
        
	signal(SIGALRM, schedule);
	alarm(TIME_SLICE);

	printf("main: swapcontext thread 0\n");
	current = 0;
        if (swapcontext(&uctx_main, &thread_save[0]) == -1)
		{ perror("swapcontext"); exit(0); }

	while (1) {
		printf("thread %d completed\n", current);
		thread_state[current] = 0;
		schedule(0);
        }
}


