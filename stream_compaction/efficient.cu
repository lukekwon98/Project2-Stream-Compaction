#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

#define blocksize 128

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void scanUpSweepSinglePass(int n, int stride, int* idata) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            int k = idx * stride;

            if (k + stride - 1 >= n) return;

            idata[k + stride - 1] += idata[k + (stride >> 1) - 1];
        }

        __global__ void scanDownSweepSinglePass(int n, int stride, int* idata) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            int k = idx * stride;

            if (k + stride - 1 >= n) return;

            int temp = idata[k + (stride >> 1) - 1];
            idata[k + (stride >> 1) - 1] = idata[k + stride - 1];
            idata[k + stride - 1] += temp;
        }

        void scanDeviceArray(int nPadded, int nPasses, int* dev_idata) {
            int numThreads = nPadded;
            for (int d = 0; d < nPasses; d++) {
                numThreads = numThreads >> 1;
                dim3 blocksPerGrid((numThreads + blocksize - 1) / blocksize);
                int stride = 1 << (d + 1);

                scanUpSweepSinglePass << <blocksPerGrid, blocksize >> > (nPadded, stride, dev_idata);
                checkCUDAError("up sweep failed");
            }

            int numDownSweepThreads = 1;
            cudaMemset(dev_idata + (nPadded - 1), 0, sizeof(int));
            for (int d = nPasses - 1; d >= 0; d--) {
                dim3 blocksPerGrid((numDownSweepThreads + blocksize - 1) / blocksize);
                int stride = 1 << (d + 1);

                scanDownSweepSinglePass << <blocksPerGrid, blocksize >> > (nPadded, stride, dev_idata);
                checkCUDAError("down sweep failed");
                numDownSweepThreads = numDownSweepThreads << 1;
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int* odata, const int* idata) {
            // TODO
            if (n <= 0) return;

            int nPasses = ilog2ceil(n);
            int nPadded = 1 << nPasses;

            int* dev_idata;

            cudaMalloc((void**)&dev_idata, nPadded * sizeof(int));
            checkCUDAError("cudaMalloc dev_idata failed!");

            cudaMemset(dev_idata, 0, nPadded * sizeof(int));
            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            timer().startGpuTimer();

            scanDeviceArray(nPadded, nPasses, dev_idata);
            
            timer().endGpuTimer();

            cudaMemcpy(odata, dev_idata, n * sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(dev_idata);
        }

        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, int* odata, const int* idata) {
            // TODO
            int* dev_idata;
            int* dev_scannedBoolList;
            int* dev_tempBoolList;
            int* dev_streamCompactResult;
            int lastScanned;
            int lastBool;

            if (n <= 0) return 0;
            int nPasses = ilog2ceil(n);
            int nPadded = 1 << nPasses;

            cudaMalloc((void**)&dev_scannedBoolList, nPadded * sizeof(int));
            checkCUDAError("cudaMalloc dev_scannedBoolList failed!");
            cudaMemset(dev_scannedBoolList, 0, nPadded * sizeof(int));

            cudaMalloc((void**)&dev_tempBoolList, nPadded * sizeof(int));
            checkCUDAError("cudaMalloc dev_tempBoolList failed!");

            cudaMalloc((void**)&dev_streamCompactResult, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_streamCompactResult failed!");

            cudaMalloc((void**)&dev_idata, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_idata failed!");
            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            timer().startGpuTimer();

            dim3 blocksPerGrid((n + blocksize - 1) / blocksize);
            Common::kernMapToBoolean<< <blocksPerGrid, blocksize >> > (n, dev_scannedBoolList, dev_idata);

            cudaMemcpy(dev_tempBoolList, dev_scannedBoolList, n * sizeof(int), cudaMemcpyDeviceToDevice);
            scanDeviceArray(nPadded, nPasses, dev_scannedBoolList);

            Common::kernScatter << <blocksPerGrid, blocksize >> > (n, dev_streamCompactResult, dev_idata, dev_tempBoolList, dev_scannedBoolList);
           
            timer().endGpuTimer();

            cudaMemcpy(&lastScanned, dev_scannedBoolList + (n - 1), sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&lastBool, dev_tempBoolList + (n - 1), sizeof(int), cudaMemcpyDeviceToHost);
            int numCompact = lastScanned + lastBool;

            cudaMemcpy(odata, dev_streamCompactResult, numCompact * sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(dev_idata);
            cudaFree(dev_scannedBoolList);
            cudaFree(dev_tempBoolList);
            cudaFree(dev_streamCompactResult);

            return numCompact;
        }
    }
}
