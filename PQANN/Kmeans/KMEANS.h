//
// Created by Kaiser Tan on 2024/4/21.
//

#ifndef PQANN_KMEANS_H
#define PQANN_KMEANS_H

#include <vector>

class KMEANS {
public:
    KMEANS(unsigned int k, unsigned int max_iter) : k(k), max_iter(max_iter) {}
    ~KMEANS() = default;

    void initializeCentroids(const std::vector<std::vector<float>> &data);

    std::vector<std::vector<float>> fit(const std::vector<std::vector<float>> &data);

    std::vector<std::vector<float>> KMEANS::fit_avx2(const std::vector<std::vector<float>> &data);

private:
    unsigned int k;
    unsigned int max_iter;
    std::vector<std::vector<float>> centroids;
};


#endif //PQANN_KMEANS_H
