//
// Created by Kaiser Tan on 2024/4/21.
//

#ifndef PQANN_CALC_H
#define PQANN_CALC_H


#include <vector>

double distance(const std::vector<float> &datapoint0, const std::vector<float> &datapoint1);

double distance_unroll(const std::vector<float> &datapoint0, const std::vector<float> &datapoint1);

double avx2_distance(const std::vector<float> &datapoint0, const std::vector<float> &datapoint1);


#endif //PQANN_CALC_H
