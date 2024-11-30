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
 * @file utils.h
 * @author Alessandro Capotondi
 * @date 27 Mar 2020
 * @brief File containing utilities functions for HPC Unimore Class
 *
 * The header define time functions and dummy workload used on the example tests.
 * 
 * @see http://algo.ing.unimo.it/people/andrea/Didattica/HPC/index.html
 */
#ifndef __UTILS_H__
#define __UTILS_H__

#include <stdarg.h>

#if defined(VERBOSE)
#define DEBUG_PRINT(x, ...) printf((x), ##__VA_ARGS__)
#else
#define DEBUG_PRINT(x, ...)
#endif

#if !defined(NTHREADS)
#define NTHREADS (4)
#endif

extern "C"
{

/**
 * @brief The function set the timestampA
 *
 * The function is used to measure elapsed time between two execution points.
 * The function start_timer() sets the starting point timestamp, while the function
 * stop_timer() sets the termination timestamp. The elapsed time, expressed in nanoseconds,
 * between the two points can be retrieved using the function elapsed_ns().
 * 
 * Example usage:
 * @code
 * start_timer(); // Point A
 * //SOME CODE HERE
 * stop_timer(); // Point B
 * printf("Elapsed time = %llu ns\n", elapsed_ns())); //Elapsed time between A and B
 * //SOME OTHER CODE HERE
 * stop_timer(); // Point C
 * printf("Elapsed time = %llu ns\n", elapsed_ns())); //Elapsed time between A and C
 * @endcode
 * 
 * @return void
 * @see start_timer()
 * @see stop_timer()
 * @see elapsed_ns()
 */
    void start_timer();

/**
 * @brief The function set the second timestamps
 *
 * The function is used to measure elapsed time between two execution points.
 * The function start_timer() sets the starting point timestamp, while the function
 * stop_timer() returns the elapsed time, expressed in nanoseconds between the last call
 * of start_timer() and the current execution point.
 * 
 * Example usage:
 * @code
 * start_timer(); // Point A
 * //SOME CODE HERE
 * stop_timer(); // Point B
 * printf("Elapsed time = %llu ns\n", elapsed_ns())); //Elapsed time between A and B
 * //SOME OTHER CODE HERE
 * stop_timer(); // Point C
 * printf("Elapsed time = %llu ns\n", elapsed_ns())); //Elapsed time between A and C
 * @endcode
 * 
 * @return void
 * @see start_timer()
 * @see stop_timer()
 * @see elapsed_ns()
 */
    void stop_timer();

/**
 * @brief Elapsed nano seconds between start_timer() and stop_timer().
 *
 * @return Elapsed nano seconds
 * @see start_timer()
 * @see stop_timer()
 */
    unsigned long long elapsed_ns();

/**
 * @brief The function init the starting point of stat measurement.
 *
 * The function is similar to start_timer().
 * 
 * @return void
 * @see start_timer
 */
    void start_stats();

/**
 * @brief The function collects the elapsed time between the current exeuction point and the 
 * last call of start_stats().
 * 
 * @return void
 */
    void collect_stats();

/**
 * @brief The function display the collected statistics.
 * @return void
 */
    void print_stats();
}    
#endif /*__UTILS_H__*/
