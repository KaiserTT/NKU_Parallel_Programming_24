//
// Created by Kaiser Tan on 2024/4/21.
//

#include "../calc/calc.h"
#include "KMEANS.h"
#include <cmath>
#include <random>
#include <limits>
#include <iostream>
#include <immintrin.h>

// 从数据集中随机选择 k 个点作为初始聚类中心
void KMEANS::initializeCentroids(const std::vector<std::vector<float>> &data) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, data.size() - 1);

    this->centroids.clear();
    for (int i = 0; i < k; i++)
        this->centroids.push_back(data[dis(gen)]);
}

// 执行 K-means 算法
std::vector<std::vector<float>> KMEANS::fit(const std::vector<std::vector<float>> &data) {
    bool converged = false;
    std::vector<int> labels(data.size());

    this->initializeCentroids(data);

    std::cout << "Centroids initialized. Start to k-means fit:" << std::endl;

    for (int it = 0; it < this->max_iter && !converged; it++) {
        bool changed = false;
        for (int i = 0; i < data.size(); i++) {
            double min_dist = std::numeric_limits<double>::max();
            for (int j = 0; j < k; j++) {
                double dist = distance_unroll(data[i], centroids[j]);
                if (dist < min_dist) {
                    min_dist = dist;
                    if (labels[i] != j) {
                        labels[i] = j;
                        changed = true;
                    }
                }
            }
        }
        // 检查是否收敛
        if (!changed) {
            converged = true;
            break;
        }

        // 更新聚类中心
        std::vector<std::vector<double>> sum(k, std::vector<double>(data[0].size(), 0.0));
        std::vector<int> count(k, 0);
        for (int i = 0; i < data.size(); i++) {
            count[labels[i]]++;
            for (int j = 0; j < data[i].size(); j++) {
                sum[labels[i]][j] += data[i][j];
            }
        }

        for (int i = 0; i < k; i++) {
            if (count[i] > 0) {
                for (int j = 0; j < sum[i].size(); j++) {
                    centroids[i][j] = sum[i][j] / count[i];
                }
            }
        }
        if (it % 50 == 0)
            std::cout << "Iteration " << it << "/" << max_iter << " finished." << std::endl;
    }

    std::cout << "K-means fit finished." << std::endl;
    return this->centroids;
}

std::vector<std::vector<float>> KMEANS::fit_avx2(const std::vector<std::vector<float>> &data) {
    bool converged = false;
    std::vector<int> labels(data.size());

    this->initializeCentroids(data);

    std::cout << "Centroids initialized. Start to k-means fit:" << std::endl;

    for (int it = 0; it < this->max_iter && !converged; it++) {
        bool changed = false;
        for (int i = 0; i < data.size(); i++) {
            double min_dist = std::numeric_limits<double>::max();
            for (int j = 0; j < k; j++) {
                double dist = distance(data[i], centroids[j]);
                if (dist < min_dist) {
                    min_dist = dist;
                    if (labels[i] != j) {
                        labels[i] = j;
                        changed = true;
                    }
                }
            }
        }
        // 检查是否收敛
        if (!changed) {
            converged = true;
            break;
        }

        // 更新聚类中心
        std::vector<std::vector<float>> sum(k, std::vector<float>(data[0].size(), 0.0));
        std::vector<int> count(k, 0);
        for (int i = 0; i < data.size(); ++i) {
            int cluster_index = labels[i];
            count[cluster_index]++;
            for (int j = 0; j < data[i].size(); j += 8) {
                __m256 data_vec = _mm256_loadu_ps(&data[i][j]);
                __m256 sum_vec = _mm256_loadu_ps(&sum[cluster_index][j]);
                sum_vec = _mm256_add_ps(sum_vec, data_vec);
                _mm256_storeu_ps(&sum[cluster_index][j], sum_vec);
            }
        }

        // 计算新的聚类中心
        for (int i = 0; i < k; ++i) {
            if (count[i] > 0) {
                for (int j = 0; j < sum[i].size(); j += 8) {
                    __m256 sum_vec = _mm256_loadu_ps(&sum[i][j]);
                    __m256 count_vec = _mm256_set1_ps(count[i]);
                    __m256 centroid_vec = _mm256_div_ps(sum_vec, count_vec);
                    _mm256_storeu_ps(&centroids[i][j], centroid_vec);
                }
            }
        }


        if (it % 50 == 0)
            std::cout << "Iteration " << it << "/" << max_iter << " finished." << std::endl;
    }

    std::cout << "K-means fit finished." << std::endl;

    return this->centroids;
}