CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Luke (Hyuk Che) Kwon
  * [LinkedIn](https://www.linkedin.com/in/hyukchekwon/), [Personal Website](https://lukekwon98.github.io/)
* Tested on: Windows 11, AMD Ryzen 5 5600X 6-Core Processor @ ~3.7GHz 16GB, Nvidia GeForce RTX 3060 (Compute Capability 8.6)

## Performance Analysis

### Block Optimization Per Custom GPU Algorithm

<img width="600" height="371" alt="Performance Per Block Size - Custom GPU Algorithms" src="https://github.com/user-attachments/assets/25be7304-699d-4da6-bed8-d7b3c38d0188" />

```
Block Size	Naive (ms)	Efficient - No Opt (ms)	Efficient - Opt (ms)
32	        9.11	    8.39	                  2.35
64	        5.58	    5.05	                  2.40
128	        5.53	    3.64	                  2.45
256	        5.57	    3.04	                  2.45
512	        5.54	    3.10	                  2.43
1024	    6.25	    3.69	                  2.48
```


### Performance Per Array Size
<img width="600" height="371" alt="Performance Per Array Size" src="https://github.com/user-attachments/assets/8f9b5ff3-5e3b-4125-a409-19f8f022f8ae" />

```
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
