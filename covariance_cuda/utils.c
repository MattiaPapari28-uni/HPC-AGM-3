/*
 * BSD 2-Clause License
 * 
 * Copyright (c) 2020, Alessandro Capotondi
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 
 * * Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimer.
 * 
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
/**
 * @file utils.c
 * @author Alessandro Capotondi
 * @date 27 Mar 2020
 * @brief File containing utilities functions for HPC Unimore Class
 *
 * Utilities for OpenMP lab.
 * 
 * @see http://algo.ing.unimo.it/people/andrea/Didattica/HPC/index.html
 */

#define _POSIX_C_SOURCE 199309L
#include <time.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <assert.h>

extern "C" {

#include "utils.h"

#define MAX_ITERATIONS 100
static struct timespec timestampA, timestampB;
static unsigned long long statistics[MAX_ITERATIONS];
static int iterations = 0;

static unsigned long long __diff_ns(struct timespec start, struct timespec end)
{
    struct timespec temp;
    if ((end.tv_nsec - start.tv_nsec) < 0)
    {
        temp.tv_sec = end.tv_sec - start.tv_sec - 1;
        temp.tv_nsec = 1000000000ULL + end.tv_nsec - start.tv_nsec;
    }
    else
    {
        temp.tv_sec = end.tv_sec - start.tv_sec;
        temp.tv_nsec = end.tv_nsec - start.tv_nsec;
    }

    return temp.tv_nsec + temp.tv_sec * 1000000000ULL;
}

void start_timer()
{
    asm volatile("" ::
                     : "memory");
    clock_gettime(CLOCK_MONOTONIC_RAW, &timestampA);
    asm volatile("" ::
                     : "memory");
}

void stop_timer()
{
    unsigned long long elapsed = 0ULL;
    asm volatile("" ::
                     : "memory");
    clock_gettime(CLOCK_MONOTONIC_RAW, &timestampB);
    asm volatile("" ::
                     : "memory");
}

unsigned long long elapsed_ns()
{
    return __diff_ns(timestampA, timestampB);
}

void start_stats()
{
    start_timer();
}

void collect_stats()
{
    assert(iterations < MAX_ITERATIONS);
    stop_timer();
    statistics[iterations++] = elapsed_ns();
}

void print_stats()
{
    unsigned long long min = ULLONG_MAX;
    unsigned long long max = 0LL;
    double average = 0.0;
    double std_deviation = 0.0;
    double sum = 0.0;

    /*  Compute the sum of all elements */
    for (int i = 0; i < iterations; i++)
    {
        if (statistics[i] > max)
            max = statistics[i];
        if (statistics[i] < min)
            min = statistics[i];
        sum = sum + statistics[i] / 1E6;
    }
    average = sum / (double)iterations;

    /*  Compute  variance  and standard deviation  */
    for (int i = 0; i < iterations; i++)
    {
        sum = sum + pow((statistics[i] / 1E6 - average), 2);
    }
    std_deviation = sqrt(sum / (double)iterations);

    printf("AvgTime\tMinTime\tMaxTime\tStdDev\n");
    printf("%.4f ms\t%.4f ms\t%.4f ms\t%.4f\n", (double)average, (double)min / 1E6, (double)max / 1E6, (double)std_deviation);
}

}
