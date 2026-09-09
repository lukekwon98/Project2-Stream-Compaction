#include <cstdio>
#include "cpu.h"

#include "common.h"

namespace StreamCompaction {
    namespace CPU {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        /**
         * CPU scan (prefix sum).
         * For performance analysis, this is supposed to be a simple for loop.
         * (Optional) For better understanding before starting moving to GPU, you can simulate your GPU scan in this function first.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            // TODO
            if (n <= 0) {
                timer().endCpuTimer();
                return;
            }

            odata[0] = 0;
            for (int k = 1; k < n; ++k) {
                odata[k] = odata[k - 1] + idata[k - 1];
            }
            timer().endCpuTimer();
        }

        void scanWithoutTimer(int n, int* odata, const int* idata) {
            // TODO
            if (n <= 0) {
                return;
            }

            odata[0] = 0;
            for (int k = 1; k < n; ++k) {
                odata[k] = odata[k - 1] + idata[k - 1];
            }
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            // TODO
            if (n <= 0) {
                timer().endCpuTimer();
                return 0;
            }

            int num = 0;

            for (int k = 0; k < n; ++k) {
                if (idata[k] != 0) {
                    odata[num] = idata[k];
                    ++num;
                }
            }

            timer().endCpuTimer();
            return num;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            // TODO
            if (n <= 0) {
                timer().endCpuTimer();
                return 0;
            }

            int* tempBoolList = new int[n]{};
            int* scannedBoolList = new int[n];
            for (int k = 0; k < n; k++) {
                if (idata[k] != 0) {
                    tempBoolList[k] = 1;
                }
                else {
                    tempBoolList[k] = 0;
                }
            }

            scanWithoutTimer(n, scannedBoolList, tempBoolList);

            int num = 0;
            for (int k = 0; k < n; k++) {
                if (tempBoolList[k] == 1) {
                    odata[scannedBoolList[k]] = idata[k];
                    num++;
                }
            }
            
            delete[] tempBoolList;
            delete[] scannedBoolList;
            timer().endCpuTimer();

            return num;
        }
    }
}
