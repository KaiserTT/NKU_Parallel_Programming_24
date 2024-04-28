//
// Created by Kaiser Tan on 2024/4/21.
//

#include <vector>
#include "calc.h"
#include <cmath>
#include <immintrin.h>

double distance(const std::vector<float> &datapoint0, const std::vector<float> &datapoint1) {
    double sum = 0.0;
    for (int i = 0; i < datapoint0.size(); i++) {
        double diff = datapoint0[i] - datapoint1[i];
        sum += diff * diff;
    }
    return sqrt(sum);
}

double distance_unroll(const std::vector<float> &datapoint0, const std::vector<float> &datapoint1) {
    double sum = 0.0;
    int size = datapoint0.size();
    int i = 0;
    for (; i < size - 8; i += 8) {
        double diff0 = datapoint0[i] - datapoint1[i];
        double diff1 = datapoint0[i + 1] - datapoint1[i + 1];
        double diff2 = datapoint0[i + 2] - datapoint1[i + 2];
        double diff3 = datapoint0[i + 3] - datapoint1[i + 3];
        double diff4 = datapoint0[i + 4] - datapoint1[i + 4];
        double diff5 = datapoint0[i + 5] - datapoint1[i + 5];
        double diff6 = datapoint0[i + 6] - datapoint1[i + 6];
        double diff7 = datapoint0[i + 7] - datapoint1[i + 7];
        sum += diff0 * diff0 + diff1 * diff1 + diff2 * diff2 + diff3 * diff3 +
               diff4 * diff4 + diff5 * diff5 + diff6 * diff6 + diff7 * diff7;
    }

    return sqrt(sum);
}

double avx2_distance(const std::vector<float> &datapoint0, const std::vector<float> &datapoint1) {
    __m256 sum_vec = _mm256_setzero_ps();
    int size = datapoint0.size();

    for (size_t i = 0; i < size; i+= 8) {
        __m256 v0 = _mm256_loadu_ps(&datapoint0[i]);
        __m256 v1 = _mm256_loadu_ps(&datapoint1[i]);
        __m256 diff = _mm256_sub_ps(v0, v1);
        sum_vec = _mm256_fmadd_ps(diff, diff, sum_vec);
    }

    __attribute__((aligned(32))) float sum_array[8];
    _mm256_store_ps(sum_array, sum_vec);
    double sum = sum_array[0] + sum_array[1] + sum_array[2] + sum_array[3] +
                 sum_array[4] + sum_array[5] + sum_array[6] + sum_array[7];

    // 处理剩余的元素
    for (size_t i = size - size % 8; i < size; ++i) {
        float diff = datapoint0[i] - datapoint1[i];
        sum += diff * diff;
    }

    return sqrt(sum);
}