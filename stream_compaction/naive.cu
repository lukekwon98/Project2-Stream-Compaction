#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

#define blocksize 128

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        // TODO: __global__
        __global__ void scanSinglePass(int n, int d, int* odata, const int* idata) {
            int k = blockIdx.x * blockDim.x + threadIdx.x;
            if (k >= n) return;
            
            int dPow = 1 << (d - 1);

            if (k >= dPow) {
                odata[k] = idata[k - dPow] + idata[k];
            }
            else {
                odata[k] = idata[k];
            }

        }

        __global__ void inclusiveToExclusive(int n, int* odata, const int* idata) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= n) return;

            if (idx == 0) {
                odata[idx] = 0;
            }
            else {
                odata[idx] = idata[idx - 1];
            }
            
            return;
        }


        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            // TODO
            if (n <= 0) return;

            int* dev_idata;
            int* dev_odata;

            cudaMalloc((void**)&dev_idata, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_idata failed!");
            cudaMalloc((void**)&dev_odata, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_odata failed!");

            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            timer().startGpuTimer();

            dim3 blocksPerGrid((n + blocksize - 1) / blocksize);
            
            int nPasses = ilog2ceil(n);
            for (int d = 1; d <= nPasses; d++) {
                scanSinglePass << <blocksPerGrid, blocksize >> > (n, d, dev_odata, dev_idata);
                std::swap(dev_idata, dev_odata);
            }

            inclusiveToExclusive << <blocksPerGrid, blocksize >> > (n, dev_odata, dev_idata);

            timer().endGpuTimer();

            cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(dev_idata);
            cudaFree(dev_odata);
        }
    }
}
