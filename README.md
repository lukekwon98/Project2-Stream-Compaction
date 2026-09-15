CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Luke (Hyuk Che) Kwon
  * [LinkedIn](https://www.linkedin.com/in/hyukchekwon/), [Personal Website](https://lukekwon98.github.io/)
* Tested on: Windows 11, AMD Ryzen 5 5600X 6-Core Processor @ ~3.7GHz 16GB, Nvidia GeForce RTX 3060 (Compute Capability 8.6)

## List of Features

* CPU Exclusive Scan
* CPU stream compaction with and without scan
* Naive CUDA exclusive scan
* Work-efficient CUDA exclusive scan
* CUDA stream compaction using map, scan, and scatter
* Thrust exclusive scan
* Thrust steam compaction using thrust::remove_if
* Support for non-power-of-two input arrays
* Extra credit: work-efficient scan optimization that launches only the number of threads required at each upper/down sweep level

## Performance Analysis

All performance tests were run in Release mode without debugging. Initial/final memory operations such as cudaMalloc and cudaMemcpy were excluded from the measured execution time.

Each result was measured as the median over 10 runs. Block sizes were independently tuned for each custom CUDA scan implementation, and the best observed block size for each implementation was used for the array size performance comparison.


### Block Optimization Per Custom GPU Algorithm
<img width="600" height="371" alt="chart (1)" src="https://github.com/user-attachments/assets/179d5158-4fab-4a1b-abf0-388eff7e4b3f" />

```text
Block Size	Naive (ms)	Efficient - No Opt (ms)	Efficient - Opt (ms)
32	        9.11	    8.39	                  2.35
64	        5.58	    5.05	                  2.40
128	        5.53	    3.64	                  2.45
256	        5.57	    3.04	                  2.45
512	        5.54	    3.10	                  2.43
1024	    6.25	    3.69	                  2.48
```

The best observed block size was 128 threads for the naive scan, 256 threads for the unoptimized work-efficient scan.

The naive implementation improved substantially from 32 to 64 threads per block, but its performance was nearly the same between 64 and 512 threads. Although 128 threads produced the lowest measured runtime of 5.53 ms, the results between 64 and 512 threads differed only by 0.05 ms.

The unoptimized work-efficient scan was noticeably more responsive to block size, as runtime decreased from 8.39 ms at 32 threads to 3.04 ms at 256 threads before increasing again at larger block sizes.

Interestingly, the optimized work-efficient implementation was relatively insensitive to block size. Its runtime varied only from 2.35ms to 2.48 ms across all tested configuration, which is a difference of only 0.13 ms. The best observed result was when the block size was 32.

One reason the optimized implementation may perform well with smaller blocks is that the number of useful operations decreases rapidly toward the root of the up-sweep and begins small during the down-sweep. Since the optimized implementation changes its launch size at each tree level, smaller block size may reduce the number of inactive threads in partially filled blocks at the narrower levels.

For all following tests, each implementation was run using the following block sizes:

| Implementation | Block Size |
|---|---:|
| Naive | 128 |
| Efficient - No Optimization | 256 |
| Efficient - Optimized | 32 |


### Performance Per Array Size
<img width="600" height="371" alt="chart" src="https://github.com/user-attachments/assets/ff181767-367c-4a5d-8ec4-f91b18adbc14" />

```text
Array Size    CPU (ms)    Naive (ms)    Efficient - No Opt (ms)    Efficient - Opt (ms)    Thrust (ms)
500,000       0.23        0.39          0.32                       0.29                     0.05
1,000,000     0.74        0.76          0.54                       0.46                     0.38
2,000,000     1.13        1.44          0.86                       0.76                     0.42
3,000,000     1.41        1.93          1.58                       1.27                     0.47
4,000,000     1.91        2.52          1.52                       1.27                     0.48
5,000,000     2.49        3.35          3.11                       2.33                     0.50
6,000,000     2.97        3.96          3.08                       2.35                     0.53
7,000,000     3.46        4.55          3.05                       2.36                     0.58
8,000,000     4.35        5.96          3.07                       2.35                     0.59
9,000,000     4.56        6.08          6.14                       4.52                     0.67
10,000,000    4.98        6.75          6.12                       4.57                     0.65
```

### Observations

#### CPU vs Navie GPU Scan

The naive GPU scan was consistently slower than the serial CPU scan for the tested sizes. At 10 million elements, the CPU scan completed in 4.98 ms while the naive GPU scan took 6.75 ms.

Although the naive implementation performs the individual operations in parallel, it performs approximately O(nlogn) total work. Each level of the scan processes approximately the entire array, and each level requires a separate kernel launch, whereas the serial CPU scan performs only O(n) instructions in a single pass.

As a result, the additional global memory traffic and repeated kernel launch overhead of the naive GPU implementation outweigh the benefit of parallel execution for the tested sizes.

#### Work-Efficient scan

The work-efficient implementation reduces the total amount of scan work from O(nlogn) to O(n) by using an up-sweep and down-sweep on a balanced tree.

The optimized work-efficient scan consequently outperformed the naive implementation for all tested array sizes and also surpassed the CPU implementation at larger input sizes. For example, at 8 million elements, the CPU implementation required 4.35 ms while the optimized work-efficient GPU scan only took 2.35 ms.

A stair-step pattern is visible in both work-efficient implementations, which occurs because non-power-of-two inputs are padded to the next power of two before performing the tree based scan. For example, 3M-4M elements will be padded to 2^22, and 5M-8M elements will be padded to 2^23 elements.

Inputs within each range therefore operate on the same padded array size and perform nearly the same amount of scan work. This also explains why execution time remains almost constant within these ranges. 



#### Performance Bottlenecks

The serial CPU scan performs only O(n) work and accesses memory sequentially, giving it good cache behavior, but it cannot exploit the large amount of parallelism available on the GPU.

The naive GPU scan exposes significant parallelism, but performs O(nlogn) work. Every scan level reads and writes a large portion of the array in global memory and requires another kernel launch. Its performance is therefore limited by both global-memory traffic and repeated launch overhead.

The work-efficient scan performs only O(n) arithmetic work, but its tree structure causes amount of available parallel work to decrease by half at every up-sweep level and increase from a single operation during the down-sweep. Therefore, near the root of the tree, there are too few useful threads to fully utilize the GPU.

The power-of-two padding required by this implementation also introduces additional work for non-power-of-two inputs. An input slightly larger than a power of two may require almost twice as much intermediate storage and tree work.

Thrust significantly outperformed the custom implementations, reaching only 0.65 ms for 10 million elements compared with 4.57 ms for the optimized work-efficient implementation. Thrust uses a substantially more optimized scan implementation than the simple global-memory tree scan implemented in this project.

## Extra Credit: Work-Efficient Thread Launch Optimization

The initial work-efficient implementation launched the same maximum-sized grid at every level of both the up-sweep and down-sweep.

However, the amount of useful parallel work changes at every tree level. During the up-sweep, the number of required operations is reduced by half each step, and the down-sweep performs the reverse progression.

Launching the maximum number of threads at every level therefore causes an increasingly large fraction of threads to immediately fail the bounds check and perform no useful work.

The optimized implementation instead computes the number of useful threads required for each individual tree level and launches only enough blocks to cover those threads.

The performance improvement is visible across all tested array sizes:

```text
Array Size    Efficient - No Opt (ms)    Efficient - Opt (ms)
5,000,000     3.11                       2.33
8,000,000     3.07                       2.35
9,000,000     6.14                       4.52
10,000,000    6.12                       4.57
```

At 10 million elements, execution time decreased from 6.12 ms to 4.57 ms, which corresponds to approximately a 25% reduction in runtime compared with the independently tuned baseline implementation.

Both versions were independently tuned for their best observed block size: 256 threads per block for the baseline and 32 threads per block for the optimized implementation. Therefore, this comparison represents the best observed performance of each implementation rather than isolating the launch-count optimization as the only changing variable.

The optimized scan is also used internally by the work-efficient stream compaction implementation.

## Thrust Analysis

Thrust's exclusive_scan substantially outperformed all custom scan implementations. At 10 million elements, Thrust completed the scan in approximately 0.65 ms compared with 4.57 ms for the optimized work-efficient implementation.

**TODO: Add Nsight Systems/Compute screenshot and analysis**

Inspect the Thrust execution timeline and describe the kernels and any allocation or memory-copy behavior visible in the profile

## Test Output - Array size: 10000000, 10 runs per test

```text
*****************************
**     SCAN CORRECTNESS    **
*****************************
    [  15   1  22  27   1  15  31   7  40  11  23  34  48 ...  23   0 ]
==== cpu scan, power-of-two ====
    [   0  15  16  38  65  66  81 112 119 159 170 193 227 ... 244908114 244908137 ]
==== cpu scan, non-power-of-two ====
    passed
==== naive scan, power-of-two ====
    passed
==== naive scan, non-power-of-two ====
    passed
==== work-efficient scan unoptimized, power-of-two ====
    passed
==== work-efficient scan unoptimized, non-power-of-two ====
    passed
==== work-efficient scan optimized, power-of-two ====
    passed
==== work-efficient scan optimized, non-power-of-two ====
    passed
==== thrust scan, power-of-two ====
    passed
==== thrust scan, non-power-of-two ====
    passed

*****************************
** COMPACTION CORRECTNESS  **
*****************************
    [   1   1   2   3   3   3   1   3   0   3   1   2   0 ...   1   0 ]
==== cpu compact without scan, power-of-two ====
    passed
==== cpu compact without scan, non-power-of-two ====
    passed
==== cpu compact with scan, power-of-two ====
    passed
==== cpu compact with scan, non-power-of-two ====
    passed
==== work-efficient compact, power-of-two ====
    passed
==== work-efficient compact, non-power-of-two ====
    passed
==== thrust remove_if, power-of-two ====
    passed
==== thrust remove_if, non-power-of-two ====
    passed

*****************************
**    SCAN PERFORMANCE     **
*****************************
CPU scan, power-of-two                                  Avg:     4.36 ms | Median:     4.33 ms | Min:     4.23 ms | Max:     4.52 ms
CPU scan, non-power-of-two                              Avg:     4.53 ms | Median:     4.52 ms | Min:     4.27 ms | Max:     5.14 ms
Naive scan, power-of-two                                Avg:     7.10 ms | Median:     6.79 ms | Min:     6.75 ms | Max:     8.58 ms
Naive scan, non-power-of-two                            Avg:     6.84 ms | Median:     6.80 ms | Min:     6.75 ms | Max:     7.28 ms
Work-efficient scan, unoptimized, power-of-two          Avg:     6.26 ms | Median:     6.28 ms | Min:     6.13 ms | Max:     6.42 ms
Work-efficient scan, unoptimized, non-power-of-two      Avg:     6.20 ms | Median:     6.15 ms | Min:     6.12 ms | Max:     6.67 ms
Work-efficient scan, optimized, power-of-two            Avg:     4.61 ms | Median:     4.53 ms | Min:     4.51 ms | Max:     5.17 ms
Work-efficient scan, optimized, non-power-of-two        Avg:     4.58 ms | Median:     4.56 ms | Min:     4.53 ms | Max:     4.74 ms
Thrust scan, power-of-two                               Avg:     0.67 ms | Median:     0.66 ms | Min:     0.66 ms | Max:     0.70 ms
Thrust scan, non-power-of-two                           Avg:     0.69 ms | Median:     0.67 ms | Min:     0.66 ms | Max:     0.84 ms

*****************************
** COMPACTION PERFORMANCE  **
*****************************
CPU compact without scan, power-of-two                  Avg:    15.77 ms | Median:    15.67 ms | Min:    15.64 ms | Max:    16.24 ms
CPU compact without scan, non-power-of-two              Avg:    16.38 ms | Median:    15.74 ms | Min:    15.63 ms | Max:    19.40 ms
CPU compact with scan, power-of-two                     Avg:    44.42 ms | Median:    44.53 ms | Min:    41.30 ms | Max:    45.87 ms
CPU compact with scan, non-power-of-two                 Avg:    45.64 ms | Median:    45.37 ms | Min:    44.26 ms | Max:    48.91 ms
Work-efficient compact, power-of-two                    Avg:     5.94 ms | Median:     5.89 ms | Min:     5.83 ms | Max:     6.39 ms
Work-efficient compact, non-power-of-two                Avg:     5.90 ms | Median:     5.88 ms | Min:     5.86 ms | Max:     6.02 ms
Thrust remove_if, power-of-two                          Avg:     0.79 ms | Median:     0.79 ms | Min:     0.70 ms | Max:     0.86 ms
Thrust remove_if, non-power-of-two                      Avg:     0.88 ms | Median:     0.89 ms | Min:     0.76 ms | Max:     0.96 ms
```
