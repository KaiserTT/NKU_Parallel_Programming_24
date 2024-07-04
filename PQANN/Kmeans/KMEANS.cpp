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
#include <omp.h>
#include <pthread.h>
#include <mpi.h>

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
                double dist = avx2_distance(data[i], centroids[j]);
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

std::vector<std::vector<float>> KMEANS::fit_openmp(const std::vector<std::vector<float>>& data, int thread_num) {
    omp_set_num_threads(thread_num);
    bool converged = false;
    std::vector<int> labels(data.size());
    initializeCentroids(data);

//     std::cout << "Centroids initialized. Start to k-means fit:" << std::endl;

    for (int it = 0; it < max_iter && !converged; it++) {
        bool changed = false;
        #pragma omp parallel for
        for (int i = 0; i < data.size(); i++) {
            double min_dist = std::numeric_limits<double>::max();
            for (int j = 0; j < k; j++) {
                double dist = avx2_distance(data[i], centroids[j]);
                if (dist < min_dist) {
                    min_dist = dist;
                    if (labels[i] != j) {
                        labels[i] = j;
                        changed = true;
                    }
                }
            }
        }

        if (!changed) {
            converged = true;
            break;
        }

        std::vector<std::vector<float>> sum(k, std::vector<float>(data[0].size(), 0.0));
        std::vector<int> count(k, 0);

        #pragma omp parallel for
        for (int i = 0; i < data.size(); i++) {
            #pragma omp atomic
            count[labels[i]]++;
            for (int j = 0; j < data[i].size(); j++) {
                #pragma omp atomic
                sum[labels[i]][j] += data[i][j];
            }
        }

        #pragma omp parallel for
        for (int i = 0; i < k; i++) {
            if (count[i] > 0) {
                for (int j = 0; j < sum[i].size(); j++) {
                    centroids[i][j] = sum[i][j] / count[i];
                }
            }
        }

//            if (it % 50 == 0)
//              std::cout << "Iteration " << it << "/" << max_iter << " finished." << std::endl;
    }

//    std::cout << "K-means fit finished." << std::endl;
    return centroids;
}

std::vector<std::vector<float>> KMEANS::fit_avx2_openmp(const std::vector<std::vector<float>> &data, int thread_num) {
    omp_set_num_threads(thread_num); // 设置 OpenMP 使用的线程数
    bool converged = false;
    std::vector<int> labels(data.size());
    this->initializeCentroids(data);

    std::cout << "Centroids initialized. Start to k-means fit:" << std::endl;

    for (int it = 0; it < this->max_iter && !converged; it++) {
        bool changed = false;

        // 使用 OpenMP 并行化数据点的标签分配
        #pragma omp parallel for reduction(+:changed)
        for (int i = 0; i < data.size(); i++) {
            double min_dist = std::numeric_limits<double>::max();
            int best_label = labels[i];
            for (int j = 0; j < k; j++) {
                double dist = avx2_distance(data[i], centroids[j]);
                if (dist < min_dist) {
                    min_dist = dist;
                    best_label = j;
                }
            }
            if (labels[i] != best_label) {
                labels[i] = best_label;
                changed = true;
            }
        }

        if (!changed) {
            converged = true;
            break;
        }

        std::vector<std::vector<float>> sum(k, std::vector<float>(data[0].size(), 0.0));
        std::vector<int> count(k, 0);

        // 使用 OpenMP 并行化质心更新
        #pragma omp parallel for
        for (int i = 0; i < data.size(); ++i) {
            int cluster_index = labels[i];
            #pragma omp atomic
            count[cluster_index]++;
            for (int j = 0; j < data[i].size(); j += 8) {
                __m256 data_vec = _mm256_loadu_ps(&data[i][j]);
                __m256 sum_vec = _mm256_loadu_ps(&sum[cluster_index][j]);
                sum_vec = _mm256_add_ps(sum_vec, data_vec);
                _mm256_storeu_ps(&sum[cluster_index][j], sum_vec);
            }
        }

        // 计算新的聚类中心
        #pragma omp parallel for
        for (int i = 0; i < k; ++i) {
            if (count[i] > 0) {
                for (int j = 0; j < sum[i].size(); j += 8) {
                    __m256 sum_vec = _mm256_loadu_ps(&sum[i][j]);
                    __m256 count_vec = _mm256_set1_ps(static_cast<float>(count[i]));
                    __m256 centroid_vec = _mm256_div_ps(sum_vec, count_vec);
                    _mm256_storeu_ps(&centroids[i][j], centroid_vec);
                }
            }
        }

        if (it % 50 == 0) {
            std::cout << "Iteration " << it << "/" << this->max_iter << " finished." << std::endl;
        }
    }

    std::cout << "K-means fit finished." << std::endl;
    return this->centroids;
}

struct ThreadData {
    const std::vector<std::vector<float>>* data;
    const std::vector<std::vector<float>>* centroids;
    std::vector<int>* labels;
    int start;
    int end;
    int k;
    bool* changed;
    pthread_mutex_t* mutex;
};

void* assign_labels(void* arg) {
    ThreadData* threadData = static_cast<ThreadData*>(arg);
    const auto& data = *threadData->data;
    const auto& centroids = *threadData->centroids;
    auto& labels = *threadData->labels;
    int k = threadData->k;
    bool local_changed = false;

    for (int i = threadData->start; i < threadData->end; i++) {
        double min_dist = std::numeric_limits<double>::max();
        for (int j = 0; j < k; j++) {
            double dist = distance(data[i], centroids[j]);
            if (dist < min_dist) {
                min_dist = dist;
                if (labels[i] != j) {
                    labels[i] = j;
                    local_changed = true;
                }
            }
        }
    }

    if (local_changed) {
        pthread_mutex_lock(threadData->mutex);
        *threadData->changed = true;
        pthread_mutex_unlock(threadData->mutex);
    }
    pthread_exit(nullptr);
}

struct UpdateData {
    const std::vector<std::vector<float>>* data;
    const std::vector<int>* labels;
    std::vector<std::vector<float>>* local_sum;
    std::vector<int>* local_count;
    int start;
    int end;
    int k;
    pthread_mutex_t* mutex;
};

void* update_centroids(void* arg) {
    UpdateData* updateData = static_cast<UpdateData*>(arg);
    const auto& data = *updateData->data;
    const auto& labels = *updateData->labels;
    auto& local_sum = *updateData->local_sum;
    auto& local_count = *updateData->local_count;
    int k = updateData->k;

    for (int i = updateData->start; i < updateData->end; i++) {
        int label = labels[i];
        local_count[label]++;
        for (size_t j = 0; j < data[i].size(); j++) {
            local_sum[label][j] += data[i][j];
        }
    }
    return nullptr;
}

void* update_centroids_avx2(void* arg) {
    UpdateData* updateData = static_cast<UpdateData*>(arg);
    const auto& data = *updateData->data;
    const auto& labels = *updateData->labels;
    auto& local_sum = *updateData->local_sum;
    auto& local_count = *updateData->local_count;
    int k = updateData->k;

    for (int i = updateData->start; i < updateData->end; i++) {
        int label = labels[i];
        local_count[label]++;
        for (size_t j = 0; j < data[i].size(); j += 8) {
            __m256 data_vec = _mm256_loadu_ps(&data[i][j]);
            __m256 sum_vec = _mm256_loadu_ps(&local_sum[label][j]);
            sum_vec = _mm256_add_ps(sum_vec, data_vec);
            _mm256_storeu_ps(&local_sum[label][j], sum_vec);
        }
    }
    return nullptr;
}

 std::vector<std::vector<float>> KMEANS::fit_pthread(const std::vector<std::vector<float>>& data, int thread_num) {
    bool converged = false;
    std::vector<int> labels(data.size(), 0);  // 初始化标签数组
    initializeCentroids(data);  // 初始化质心

    // std::cout << "Centroids initialized. Start to k-means fit:" << std::endl;

    int num_threads = thread_num;  // 使用硬件并发数作为线程数
    pthread_t threads[num_threads];
    pthread_mutex_t mutex;
    pthread_mutex_init(&mutex, nullptr);

    bool changed = false;  // 用于追踪任何一次迭代中标签是否改变

    for (int it = 0; it < max_iter && !converged; it++) {
        changed = false;
        std::vector<ThreadData> threadData(num_threads);

        int chunk_size = data.size() / num_threads;
        for (int t = 0; t < num_threads; t++) {
            int start = t * chunk_size;
            int end = (t == num_threads - 1) ? data.size() : start + chunk_size;
            threadData[t] = {&data, &centroids, &labels, start, end, k, &changed, &mutex};

            if (pthread_create(&threads[t], nullptr, assign_labels, &threadData[t]) != 0) {
                std::cerr << "Error creating thread " << t << std::endl;
                exit(1);
            }
        }

        for (int t = 0; t < num_threads; t++) {
            pthread_join(threads[t], nullptr);
        }

        if (!changed) {
            converged = true;
            break;
        }

        // 准备局部累积的数据结构
        std::vector<std::vector<std::vector<float>>> all_sum(num_threads, std::vector<std::vector<float>>(k, std::vector<float>(data[0].size(), 0.0)));
        std::vector<std::vector<int>> all_count(num_threads, std::vector<int>(k, 0));

        std::vector<UpdateData> updateData(num_threads);
        for (int t = 0; t < num_threads; t++) {
            updateData[t] = {&data, &labels, &all_sum[t], &all_count[t], threadData[t].start, threadData[t].end, k, &mutex};
            if (pthread_create(&threads[t], nullptr, update_centroids, &updateData[t]) != 0) {
                std::cerr << "Error creating thread " << t << std::endl;
                exit(1);
            }
        }

        for (int t = 0; t < num_threads; t++) {
            pthread_join(threads[t], nullptr);
        }

        // 合并所有线程的结果
        std::vector<std::vector<float>> sum(k, std::vector<float>(data[0].size(), 0.0));
        std::vector<int> count(k, 0);
        for (int i = 0; i < num_threads; i++) {
            for (int j = 0; j < k; j++) {
                count[j] += all_count[i][j];
                for (size_t d = 0; d < data[0].size(); d++) {
                    sum[j][d] += all_sum[i][j][d];
                }
            }
        }

        // 更新全局质心
        for (int i = 0; i < k; i++) {
            if (count[i] > 0) {
                for (size_t d = 0; d < sum[i].size(); d++) {
                    centroids[i][d] = sum[i][d] / count[i];
                }
            }
        }

        //if (it % 10 == 0)  // 每10次迭代输出一次进度
          //  std::cout << "Iteration " << it << " completed." << std::endl;
    }

    pthread_mutex_destroy(&mutex);
    // std::cout << "K-means fit finished." << std::endl;
    return centroids;
}

std::vector<std::vector<float>> KMEANS::fit_mpi(const std::vector<std::vector<float>>& data) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n = data.size();
    int local_n = n / size;
    int dim = data[0].size();

    std::cout << "Process " << rank << ": Starting fit_mpi with local data size " << local_n << std::endl;

    // 手动打包数据
    std::vector<float> flatten_data(n * dim);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < dim; j++) {
            flatten_data[i * dim + j] = data[i][j];
        }
    }

    std::vector<float> local_flatten_data(local_n * dim);
    MPI_Scatter(flatten_data.data(), local_n * dim, MPI_FLOAT,
                local_flatten_data.data(), local_n * dim, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // 解包到 local_data
    std::vector<std::vector<float>> local_data(local_n, std::vector<float>(dim));
    for (int i = 0; i < local_n; i++) {
        for (int j = 0; j < dim; j++) {
            local_data[i][j] = local_flatten_data[i * dim + j];
        }
    }

    bool converged = false;
    std::vector<int> labels(n);
    std::vector<int> local_labels(local_n);

    std::vector<float> flatten_centroids(k * dim);
    if (rank == 0) {
        this->initializeCentroids(data);
        // 打包 centroids
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < dim; j++) {
                flatten_centroids[i * dim + j] = centroids[i][j];
            }
        }
    }
    MPI_Bcast(flatten_centroids.data(), k * dim, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // 解包 centroids
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < dim; j++) {
            centroids[i][j] = flatten_centroids[i * dim + j];
        }
    }

    for (int it = 0; it < this->max_iter && !converged; it++) {
        std::cout << "Process " << rank << ": Iteration " << it << std::endl;
        for (int i = 0; i < local_n; i++) {
            double min_dist = std::numeric_limits<double>::max();
            for (int j = 0; j < k; j++) {
                double dist = distance(local_data[i], centroids[j]);
                if (dist < min_dist) {
                    min_dist = dist;
                    local_labels[i] = j;
                }
            }
        }

        MPI_Allgather(local_labels.data(), local_n, MPI_INT,
                      labels.data(), local_n, MPI_INT, MPI_COMM_WORLD);

        bool changed = false;
        for (int i = 0; i < n; i++) {
            if (labels[i] != local_labels[i]) {
                changed = true;
                break;
            }
        }
        MPI_Allreduce(&changed, &converged, 1, MPI_C_BOOL, MPI_LOR, MPI_COMM_WORLD);
        converged = !converged;

        if (converged) break;

        std::vector<std::vector<float>> local_sum(k, std::vector<float>(dim, 0.0));
        std::vector<int> local_count(k, 0);
        for (int i = 0; i < local_n; i++) {
            local_count[local_labels[i]]++;
            for (int j = 0; j < dim; j++) {
                local_sum[local_labels[i]][j] += local_data[i][j];
            }
        }

        std::vector<float> flatten_sum(k * dim);
        // 打包 local_sum
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < dim; j++) {
                flatten_sum[i * dim + j] = local_sum[i][j];
            }
        }

        std::vector<float> sum_result(k * dim);
        MPI_Allreduce(flatten_sum.data(), sum_result.data(), k * dim, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

        std::vector<std::vector<float>> sum(k, std::vector<float>(dim));
        // 解包 sum_result
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < dim; j++) {
                sum[i][j] = sum_result[i * dim + j];
            }
        }

        std::vector<int> count(k);
        MPI_Allreduce(local_count.data(), count.data(), k, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

        for (int i = 0; i < k; i++) {
            if (count[i] > 0) {
                for (int j = 0; j < dim; j++) {
                    centroids[i][j] = sum[i][j] / count[i];
                }
            }
        }
    }

    std::cout << "Process " << rank << ": Finished fit_mpi" << std::endl;

    return this->centroids;
}
