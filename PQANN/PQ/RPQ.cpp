//
// Created by Kaiser Tan on 2024/4/23.
//

#include "RPQ.h"
#include "../calc/calc.h"
#include <limits>
#include <immintrin.h>
#include <pthread.h>
#include <omp.h>

void RPQ::train(const SiftData<float> &siftData) {
    PQ_Codebooks codebooks0 = pq.train(siftData);
    this->codebooks[0] = codebooks0;
    std::cout << "First RPQ codebooks finished. codebooks0: (" << codebooks0.size() << ", " << codebooks0[0].size() << ", " << codebooks0[0][0].size() << ")" << std::endl;
    std::cout << "======================================" << std::endl;
    PQ_Index index0 = pq.buildIndex(siftData);
    this->index[0] = index0;
    std::cout << "First RPQ index finished. index0: (" << index0.size() << ", " << index0[0].size() << ")" << std::endl;
    SiftData<float> residuals0 = calc_residuals(siftData, index0, codebooks0);
    std::cout << "======================================" << std::endl;
    std::cout << "First RPQ residuals finished. residuals0: (" << residuals0.get_num() << ", " << residuals0.get_dim() << ")" << std::endl;
    std::cout << "======================================" << std::endl;

    PQ_Codebooks codebooks1 = pq.train(residuals0);
    this->codebooks[1] = codebooks1;
    std::cout << "Second RPQ codebooks finished. codebooks1: (" << codebooks1.size() << ", " << codebooks1[0].size() << ", " << codebooks1[0][0].size() << ")" << std::endl;
    std::cout << "======================================" << std::endl;
    PQ_Index index1 = pq.buildIndex(residuals0);
    this->index[1] = index1;
    std::cout << "Second RPQ index finished. index1: (" << index1.size() << ", " << index1[0].size() << ")" << std::endl;
    std::cout << "======================================" << std::endl;
    SiftData<float> residuals1 = calc_residuals(residuals0, index1, codebooks1);
    std::cout << "Second RPQ residuals finished. residuals1: (" << residuals1.get_num() << ", " << residuals1.get_dim() << ")" << std::endl;
    std::cout << "======================================" << std::endl;

    PQ_Codebooks codebooks2 = pq.train(residuals1);
    this->codebooks[2] = codebooks2;
    std::cout << "Third RPQ codebooks finished. codebooks2: (" << codebooks2.size() << ", " << codebooks2[0].size() << ", " << codebooks2[0][0].size() << ")" << std::endl;
    std::cout << "======================================" << std::endl;
    PQ_Index index2 = pq.buildIndex(residuals1);
    this->index[2] = index2;
    std::cout << "Third RPQ index finished. index2: (" << index2.size() << ", " << index2[0].size() << ")" << std::endl;
    std::cout << "======================================" << std::endl;

    std::cout << "RPQ training finished." << std::endl;
    std::cout << "codebooks: (" << codebooks.size() << ", " << codebooks[0].size() << ", " << codebooks[0][0].size() << ", " << codebooks[0][0][0].size() << ")" << std::endl;
    std::cout << "index: (" << index.size() << ", " << index[0].size() << ", " << index[0][0].size() << ")" << std::endl;
    std::cout << "======================================" << std::endl;
}

SiftData<float> RPQ::calc_residuals(const SiftData<float> &siftData, const PQ_Index &index, const PQ_Codebooks &codebooks) {
    unsigned int subspace_dim = siftData.get_dim() / pq.get_subspace_num();
    unsigned int subspace_num = pq.get_subspace_num();
    SiftData<float> residuals(siftData.get_num(), siftData.get_dim());
    for (int i = 0; i < siftData.get_num(); i++) {
        for (int j = 0; j < subspace_num; j++) {
            std::vector<float> subdatapoint(subspace_dim);
            for (int k = 0; k < subspace_dim; k++) {
                subdatapoint[k] = siftData.data[i][j * subspace_dim + k];
            }
            std::vector<float> residual = calc_datapoint_residuals_avx2(subdatapoint, codebooks[j][index[i][j]]);
            for (int k = 0; k < subspace_dim; k++) {
                residuals.data[i][j * subspace_dim + k] = residual[k];
            }
        }
    }
    return residuals;
}

struct ThreadData {
    RPQ* rpq;
    const SiftData<float>* siftData;
    const PQ_Index* index;
    const PQ_Codebooks* codebooks;
    SiftData<float>* residuals;
    int start_idx, end_idx;
};

void* threadFunc(void* arg) {
    ThreadData* data = (ThreadData*)arg;
    for (int i = data->start_idx; i < data->end_idx; i++) {
        for (int j = 0; j < data->rpq->get_subspace_num(); j++) {
            std::vector<float> subdatapoint(data->siftData->get_dim() / data->rpq->get_subspace_num());
            for (int k = 0; k < subdatapoint.size(); k++) {
                subdatapoint[k] = data->siftData->data[i][j * subdatapoint.size() + k];
            }
            std::vector<float> residual = data->rpq->calc_datapoint_residuals(subdatapoint, (*data->codebooks)[j][(*data->index)[i][j]]);
            for (int k = 0; k < subdatapoint.size(); k++) {
                data->residuals->data[i][j * subdatapoint.size() + k] = residual[k];
            }
        }
    }
    return nullptr;
}

SiftData<float> RPQ::calc_residuals_pthread(const SiftData<float>& siftData, const PQ_Index& index, const PQ_Codebooks& codebooks, int thread_num) {
    int numThreads = thread_num; // 可根据硬件调整线程数
    std::vector<pthread_t> threads(numThreads);
    std::vector<ThreadData> threadData(numThreads);
    SiftData<float> residuals(siftData.get_num(), siftData.get_dim());

    int numPerThread = siftData.get_num() / numThreads;
    for (int i = 0; i < numThreads; i++) {
        threadData[i] = {this, &siftData, &index, &codebooks, &residuals, i * numPerThread, (i + 1) * numPerThread};
        if (i == numThreads - 1) {
            threadData[i].end_idx = siftData.get_num(); // 确保最后一个线程处理所有剩余的元素
        }
        pthread_create(&threads[i], nullptr, threadFunc, &threadData[i]);
    }

    for (int i = 0; i < numThreads; i++) {
        pthread_join(threads[i], nullptr);
    }
    return residuals;
}

SiftData<float> RPQ::calc_residuals_openmp(const SiftData<float> &siftData, const PQ_Index &index, const PQ_Codebooks &codebooks, int thread_num) {
    unsigned int subspace_dim = siftData.get_dim() / pq.get_subspace_num();
    unsigned int subspace_num = pq.get_subspace_num();
    SiftData<float> residuals(siftData.get_num(), siftData.get_dim());
    omp_set_num_threads(thread_num);
    #pragma omp parallel for
    for (int i = 0; i < siftData.get_num(); i++) {
        for (int j = 0; j < subspace_num; j++) {
            std::vector<float> subdatapoint(subspace_dim);
            for (int k = 0; k < subspace_dim; k++) {
                subdatapoint[k] = siftData.data[i][j * subspace_dim + k];
            }
            std::vector<float> residual = calc_datapoint_residuals(subdatapoint, codebooks[j][index[i][j]]);
            for (int k = 0; k < subspace_dim; k++) {
                residuals.data[i][j * subspace_dim + k] = residual[k];
            }
        }
    }
    return residuals;
}

std::vector<float> RPQ::calc_datapoint_residuals(const std::vector<float> &datapoint0, const std::vector<float> &datapoint1) {
    std::vector<float> residual(datapoint0.size());
    for (size_t i = 0; i < datapoint0.size(); i++) {
        residual[i] = datapoint0[i] - datapoint1[i];  // 计算残差
    }
    return residual;
}

unsigned int RPQ::asymmetric_query(const std::vector<float> &querypoint) {
    unsigned int best_idx = -1;
    double min_dist = (std::numeric_limits<double>::max)();
    unsigned int subspace_dim = querypoint.size() / pq.get_subspace_num();

    for (int i = 0; i < this->index[0].size(); i++) {
        double dist = 0.0;
        for (int layer = 0; layer < this->layers; layer++) {
            for (int sub = 0; sub < pq.get_subspace_num(); sub++) {
                std::vector<float> subquerypoint(subspace_dim);
                for (int k = 0; k < subspace_dim; k++) {
                    subquerypoint[k] = querypoint[sub * subspace_dim + k];
                }
                int centroid = index[layer][i][sub];
                dist += distance_unroll(subquerypoint, codebooks[layer][sub][centroid]);
            }
        }
        if (dist < min_dist) {
            min_dist = dist;
            best_idx = i;
        }
    }
    std::cout << "asymmetric query finished. best_idx: " << best_idx << " distance=" << min_dist << std::endl;
    std::cout << "======================================" << std::endl;
    return best_idx;
}

void RPQ::save_codebooks(const std::string &filename) {
    std::ofstream out(filename, std::ios::binary | std::ios::trunc);
    if (!out.is_open()) {
        std::cerr << "Failed to open file for writing: " << filename << std::endl;
        return;
    }

    int dimensions[4] = {
            static_cast<int>(codebooks.size()),
            static_cast<int>(codebooks.empty() ? 0 : codebooks[0].size()),
            static_cast<int>(codebooks.empty() || codebooks[0].empty() ? 0 : codebooks[0][0].size()),
            static_cast<int>(codebooks.empty() || codebooks[0].empty() || codebooks[0][0].empty() ? 0 : codebooks[0][0][0].size())
    };

    out.write(reinterpret_cast<char*>(dimensions), sizeof(dimensions));
    for (const auto& thirdDim : codebooks) {
        for (const auto& matrix : thirdDim) {
            for (const auto& row : matrix) {
                out.write(reinterpret_cast<const char*>(row.data()), row.size() * sizeof(float));
            }
        }
    }

    out.close();
    std::cout << "RPQ saved. " << "codebooks: ("
              << dimensions[0] << ", "
              << dimensions[1] << ", "
              << dimensions[2] << ", "
              << dimensions[3] << ")" << std::endl;
}

void RPQ::read_codebooks(const std::string &filename) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) {
        std::cerr << "Cannot open file for reading: " << filename << std::endl;
        return;
    }

    // Read dimensions information
    int dimensions[4];
    in.read(reinterpret_cast<char*>(dimensions), sizeof(dimensions));
    if (in.gcount() != sizeof(dimensions)) {
        std::cerr << "Failed to read dimensions from file: " << filename << std::endl;
        return;
    }

    // Extract dimensions for each level
    int outerSize = dimensions[0];
    int thirdSize = dimensions[1];
    int secondSize = dimensions[2];
    int innerSize = dimensions[3];

    // Resize the four-dimensional vector accordingly
    codebooks.resize(outerSize);
    for (int i = 0; i < outerSize; ++i) {
        codebooks[i].resize(thirdSize);
        for (int j = 0; j < thirdSize; ++j) {
            codebooks[i][j].resize(secondSize);
            for (int k = 0; k < secondSize; ++k) {
                codebooks[i][j][k].resize(innerSize);
                in.read(reinterpret_cast<char*>(codebooks[i][j][k].data()), innerSize * sizeof(float));
                if (in.gcount() != innerSize * sizeof(float)) {
                    std::cerr << "Failed to read data properly from file: " << filename << std::endl;
                    return;
                }
            }
        }
    }

    in.close();
    std::cout << "RPQ read. " << "codebooks: ("
              << outerSize << ", " << thirdSize << ", " << secondSize << ", " << innerSize << ")" << std::endl;
}

void RPQ::save_index(const std::string &filename) {
    std::ofstream out(filename, std::ios::binary | std::ios::trunc);
    if (!out.is_open()) {
        std::cerr << "Failed to open file for writing: " << filename << std::endl;
        return;
    }
    int dimensions[3] = {static_cast<int>(index.size()),
                         static_cast<int>(index.empty() ? 0 : index[0].size()),
                         static_cast<int>(index.empty() || index[0].empty() ? 0 : index[0][0].size())};
    out.write(reinterpret_cast<char*>(dimensions), sizeof(dimensions));

    for (const auto& matrix : index) {
        for (const auto& row : matrix) {
            out.write(reinterpret_cast<const char*>(row.data()), row.size() * sizeof(int));
        }
    }

    out.close();

    std::cout << "RPQ saved. " << "index: (" << index.size() << ", " << index[0].size() << ", " << index[0][0].size() << ")" << std::endl;
}

void RPQ::read_index(const std::string &filename) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) {
        std::cerr << "Cannot open file for reading: " << filename << std::endl;
        return;
    }
    int dimensions[3];
    in.read(reinterpret_cast<char*>(dimensions), sizeof(dimensions));
    if (in.gcount() != sizeof(dimensions)) {
        std::cerr << "Failed to read dimensions from file: " << filename << std::endl;
        return;
    }

    int outerSize = dimensions[0];
    int middleSize = dimensions[1];
    int innerSize = dimensions[2];

    index.resize(outerSize);
    for (int i = 0; i < outerSize; ++i) {
        index[i].resize(middleSize);
        for (int j = 0; j < middleSize; ++j) {
            index[i][j].resize(innerSize);
            in.read(reinterpret_cast<char*>(index[i][j].data()), innerSize * sizeof(int));
            if (in.gcount() != innerSize * sizeof(int)) {
                std::cerr << "Failed to read data properly from file: " << filename << std::endl;
                return;
            }
        }
    }

    in.close();

    std::cout << "RPQ read. " << "index: (" << index.size() << ", " << index[0].size() << ", " << index[0][0].size() << ")" << std::endl;
}

std::vector<float> RPQ::calc_datapoint_residuals_avx2(const std::vector<float> &datapoint0, const std::vector<float> &datapoint1) {
    std::vector<float> residual(datapoint0.size());
    __m256 sum_vec = _mm256_setzero_ps();
    int size = datapoint0.size();

    for (size_t i = 0; i < size; i+= 8) {
        __m256 v0 = _mm256_loadu_ps(&datapoint0[i]);
        __m256 v1 = _mm256_loadu_ps(&datapoint1[i]);
        __m256 diff = _mm256_sub_ps(v0, v1);
        _mm256_storeu_ps(&residual[i], diff);
    }

    return residual;
}
