#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include <thrust/remove.h>
#include <thrust/functional.h>
#include "common.h"
#include "thrust.h"

namespace StreamCompaction {
    namespace Thrust {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            // TODO use `thrust::exclusive_scan`
            // example: for device_vectors dv_in and dv_out:
            // thrust::exclusive_scan(dv_in.begin(), dv_in.end(), dv_out.begin());
            if (n <= 0) return;

            thrust::host_vector<int> thrust_host_in(idata, idata + n);
            thrust::device_vector<int> thrust_dev_in(thrust_host_in);
            thrust::device_vector<int> thrust_dev_out(n);

            timer().startGpuTimer();
            thrust::exclusive_scan(thrust_dev_in.begin(), thrust_dev_in.end(), thrust_dev_out.begin());
            timer().endGpuTimer();

            thrust::copy(thrust_dev_out.begin(), thrust_dev_out.end(), odata);
        }

        int thrustStreamCompaction(int n, int *odata, const int* idata) {
            if (n <= 0) return 0;

            thrust::host_vector<int> thrust_host_in(idata, idata + n);
            thrust::device_vector<int> thrust_dev_in(thrust_host_in);

            timer().startGpuTimer();
            auto streamCompactionEnd = thrust::remove_if(thrust_dev_in.begin(), thrust_dev_in.end(), thrust::logical_not<int>());
            timer().endGpuTimer();

            int numStreamCompact = streamCompactionEnd - thrust_dev_in.begin();
            thrust::copy(thrust_dev_in.begin(), streamCompactionEnd, odata);

            return numStreamCompact;
        }
    }
}
