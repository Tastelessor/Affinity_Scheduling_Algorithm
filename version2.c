#include <stdio.h>
#include <math.h>

#define N 1729
#define reps 1000

#include <omp.h>

double a[N][N], b[N][N], c[N];
int jmax[N];

// Number of threads
int thread_number;

// Structure for information about each thread
struct Thread {
    int cur; // current index
    int stop; // current destination end index
    int end; // the maximum index for its iterations
};

void init1(void);

void init2(void);

void runloop(int);

void loop1chunk(int, int);

void loop2chunk(int, int);

void valid1(void);

void valid2(void);

int helpMostLoadedThread(struct Thread[], int *, int *, int *);

int main(int argc, char *argv[]) {

    double start1, start2, end1, end2;
    int r;

    // Obtain the number of threads
#pragma omp parallel default(none) shared(thread_number)
    {
#pragma omp single
        {
            thread_number = omp_get_max_threads();
        }
    }

    init1();

    start1 = omp_get_wtime();

    for (r = 0; r < reps; r++) {
        runloop(1);
    }

    end1 = omp_get_wtime();

    valid1();

    printf("Total time for %d reps of loop 1 = %f\n", reps, (float) (end1 - start1));


    init2();

    start2 = omp_get_wtime();

    for (r = 0; r < reps; r++) {
        runloop(2);
    }

    end2 = omp_get_wtime();

    valid2();

    printf("Total time for %d reps of loop 2 = %f\n", reps, (float) (end2 - start2));

}

void init1(void) {
    int i, j;

    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            a[i][j] = 0.0;
            b[i][j] = 1.618 * (i + j);
        }
    }

}

void init2(void) {
    int i, j, expr;

    for (i = 0; i < N; i++) {
        expr = i % (4 * (i / 60) + 1);
        if (expr == 0) {
            jmax[i] = N / 2;
        } else {
            jmax[i] = 1;
        }
        c[i] = 0.0;
    }

    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            b[i][j] = (double) (i * j + 1) / (double) (N * N);
        }
    }

}

void runloop(int loopid) {
    // Declare array of structures to record state of each thread
    struct Thread threads[thread_number];

    /*
     * In an endless loop, every thread will try to finish their chunks. Once a thread finishes
     * its local set, it tries to help another thread by invoking helpMostLoadedThread(). If there
     * is no thread that needs help, break.
     * Important features:
     * A thread must finishes its local set before helping others
     * Once a thread is assigned with a chunk, it will finish all iterations in that chunk by itself
     */
#pragma omp parallel default(none) shared(loopid, threads, thread_number)
    {
        // Obtain configurations of each thread
        int myid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();
        int ipt = (int) ceil((double) N / (double) nthreads);
        int help_id = myid;
        int chunk = 1;

        // Initialize the array of threads structures
        threads[myid].cur = myid * ipt;
        threads[myid].stop = threads[myid].cur + (int) ceil((double) ipt / (double) thread_number);
        threads[myid].end = (myid + 1) * ipt > N ? N : (myid + 1) * ipt;
        if (threads[myid].stop > threads[myid].end)
            threads[myid].stop = threads[myid].end;
        #pragma omp barrier

        // Endless loop, break on if no more thread needs help
        while (1 && help_id != -1) {
            switch (loopid) {
                case 1:
                    loop1chunk(threads[myid].cur, threads[myid].stop);
                    break;
                case 2:
                    loop2chunk(threads[myid].cur, threads[myid].stop);
                    break;
            }

            // Try to finish its local set
#pragma omp critical
            {
                chunk = (threads[myid].end - threads[myid].stop) / thread_number;
                threads[myid].cur = threads[myid].stop;
                threads[myid].stop += chunk == 0 ? 1 : chunk;

                // Check if the thread finishes its local set
                if (threads[myid].cur == threads[myid].end) {
                    /*
                     * Threads may finish their local set at the same time. Thus, to avoid race
                     * condition, the section of distributing helping task must be critical
                     */

                    // Obtain id of the thread with most remaining iterations
                    help_id = helpMostLoadedThread(threads, &threads[myid].cur, &threads[myid].stop, &threads[myid].end);
                    // Break if no thread needs help
                    if (help_id >= 0)
                        // Reduce the workload of the thread that is helped
                        threads[help_id].end -= threads[myid].stop - threads[myid].cur;
                }
            }
        }
    }
}

int helpMostLoadedThread(struct Thread threads[], int *cur, int *stop, int *end) {
    int most_loaded_id = -1;
    int largest_fraction = 0;
    int remain;

    for (int i = 0; i < thread_number; i++) {
        if (threads[i].stop >= threads[i].end)
            continue;
        remain = threads[i].end - threads[i].stop;
        if (remain == 0)
            remain = 1;

        if (remain > largest_fraction) {
            most_loaded_id = i;
            largest_fraction = remain;
        }
    }
    *cur = threads[most_loaded_id].end - largest_fraction;
    *stop = threads[most_loaded_id].end;
    *end = threads[most_loaded_id].end;
    return most_loaded_id;
}

void loop1chunk(int lo, int hi) {
    int i, j;

    for (i = lo; i < hi; i++) {
        for (j = N - 1; j > i; j--) {
            a[i][j] += cos(b[i][j]);
        }
    }

}


void loop2chunk(int lo, int hi) {
    int i, j, k;
    double rN2;

    rN2 = 1.0 / (double) (N * N);

    for (i = lo; i < hi; i++) {
        for (j = 0; j < jmax[i]; j++) {
            for (k = 0; k < j; k++) {
                c[i] += (k + 1) * log(b[i][j]) * rN2;
            }
        }
    }

}


void valid1(void) {
    int i, j;
    double suma;

    suma = 0.0;
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            suma += a[i][j];
        }
    }
    printf("Loop 1 check: Sum of a is %lf\n", suma);

}


void valid2(void) {
    int i;
    double sumc;

    sumc = 0.0;
    for (i = 0; i < N; i++) {
        sumc += c[i];
    }
    printf("Loop 2 check: Sum of c is %f\n", sumc);
}


