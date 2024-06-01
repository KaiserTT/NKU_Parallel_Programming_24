//
// Created by Kaiser Tan on 2024/4/21.
//

#ifndef PQANN_KMEANS_H
#define PQANN_KMEANS_H

#include <vector>
#include <pthread.h>
#include <stdexcept>
#include <iostream>

class KMEANS {
public:
    KMEANS(unsigned int k, unsigned int max_iter) : k(k), max_iter(max_iter) {
        if (pthread_mutex_init(&mutex, nullptr) != 0) {
            throw std::runtime_error("Mutex init failed");
        }
    }
    ~KMEANS() {
        pthread_mutex_destroy(&mutex);
    }

    void initializeCentroids(const std::vector<std::vector<float>> &data);

    std::vector<std::vector<float>> fit(const std::vector<std::vector<float>> &data);

    std::vector<std::vector<float>> fit_avx2(const std::vector<std::vector<float>> &data);

    std::vector<std::vector<float>> fit_openmp(const std::vector<std::vector<float>>& data, int thread_num);

    std::vector<std::vector<float>> fit_avx2_openmp(const std::vector<std::vector<float>> &data, int thread_num);

    std::vector<std::vector<float>> fit_pthread(const std::vector<std::vector<float>>& data, int thread_num);

    void test() {
        for (int i = 0; i < 10; i++) {
            for (int j = 0; j < 10; j++) {
                std::cout << centroids[i][j] << " ";
            }
            std::cout << std::endl;
        }
    }

private:
    unsigned int k;
    unsigned int max_iter;
    std::vector<std::vector<float>> centroids;
    pthread_mutex_t mutex;

};


#endif //PQANN_KMEANS_H
