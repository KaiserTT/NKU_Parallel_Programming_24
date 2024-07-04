//
// Created by Kaiser Tan on 2024/4/21.
//

#include "PQ.h"
#include "../Kmeans/KMEANS.h"
#include "../calc/calc.h"
#include <iostream>
#include <fstream>
#include <limits>
#include <windows.h>
#include <pthread.h>
#include <omp.h>
#include <mpi.h>


struct Quantize_Thread_Data {
    const std::vector<float> *datapoint;
    PQ_Codebooks *codebooks;
    std::vector<int> *codes;
    int subspace_idx;
    unsigned int subspace_dim;
    int centroid_num;
};

void* quantize_subspace(void* arg) {
    Quantize_Thread_Data *data = static_cast<Quantize_Thread_Data*>(arg);
    int subspace_idx = data->subspace_idx;
    unsigned int subspace_dim = data->subspace_dim;
    const std::vector<float> &datapoint = *data->datapoint;
    PQ_Codebooks &codebooks = *data->codebooks;
    std::vector<int> &codes = *data->codes;
    int centroid_num = data->centroid_num;

    double min_dist = std::numeric_limits<double>::max();
    int best_idx = -1;
    std::vector<float> subdatapoint(subspace_dim);
    for (int j = 0; j < centroid_num; j++) {
        for (unsigned int k = 0; k < subspace_dim; k++) {
            subdatapoint[k] = datapoint[subspace_idx * subspace_dim + k];
        }
        double dist = distance(subdatapoint, codebooks[subspace_idx][j]);
        if (dist < min_dist) {
            min_dist = dist;
            best_idx = j;
        }
    }
    codes[subspace_idx] = best_idx;
    pthread_exit(NULL);
}

std::vector<int> PQ::quantize_pthread(const std::vector<float> &datapoint) {
    unsigned int subspace_dim = datapoint.size() / subspace_num;
    std::vector<int> codes(subspace_num);
    pthread_t threads[subspace_num];
    Quantize_Thread_Data thread_data[subspace_num];

    for (unsigned int i = 0; i < subspace_num; i++) {
        thread_data[i].datapoint = &datapoint;
        thread_data[i].codebooks = &this->codebooks;
        thread_data[i].codes = &codes;
        thread_data[i].subspace_idx = i;
        thread_data[i].subspace_dim = subspace_dim;
        thread_data[i].centroid_num = centroid_num;

        pthread_create(&threads[i], NULL, quantize_subspace, (void*)&thread_data[i]);
    }

    for (unsigned int i = 0; i < subspace_num; i++) {
        pthread_join(threads[i], NULL);
    }

    return codes;
}

std::vector<int> PQ::quantize_openmp(const std::vector<float>& datapoint) {
    unsigned int subspace_dim = datapoint.size() / subspace_num;
    std::vector<int> codes(subspace_num);

    omp_set_num_threads(4);

    #pragma omp parallel for
    for (int i = 0; i < subspace_num; i++) {
        double min_dist = std::numeric_limits<double>::max();
        int best_idx = -1;
        std::vector<float> subdatapoint(subspace_dim);

        for (int j = 0; j < centroid_num; j++) {
            for (unsigned int k = 0; k < subspace_dim; k++) {
                subdatapoint[k] = datapoint[i * subspace_dim + k];
            }
            double dist = distance(subdatapoint, codebooks[i][j]);
            if (dist < min_dist) {
                min_dist = dist;
                best_idx = j;
            }
        }
        codes[i] = best_idx;
    }

    return codes;
}


std::vector<int> PQ::quantize(const std::vector<float> &datapoint) {
    unsigned int subspace_dim = datapoint.size() / subspace_num;
    std::vector<int> codes(subspace_num);

    for (int i = 0; i < subspace_num; i++) {
        double min_dist = (std::numeric_limits<double>::max)();
        int best_idx = -1;
        for (int j = 0; j < centroid_num; j++) {
            std::vector<float> subdatapoint(subspace_dim);
            for (int k = 0; k < subspace_dim; k++) {
                subdatapoint[k] = datapoint[i * subspace_dim + k];
            }
            double dist = distance(subdatapoint, this->codebooks[i][j]);
            if (dist < min_dist) {
                min_dist = dist;
                best_idx = j;
            }
        }
        codes[i] = best_idx;
    }
    return codes;
}




int PQ::asymmetric_query(const std::vector<float> &querypoint) {
    unsigned int subspace_dim = querypoint.size() / subspace_num;
    std::vector<int> codes(subspace_num);
    int best_idx = -1;
    double min_dist = (std::numeric_limits<double>::max)();

    for (int i = 0; i < index.size(); i++) {
        double dist = 0.0;
        for (int j = 0; j < subspace_num; j++) {
            std::vector<float> subquerypoint(subspace_dim);
            for (int k = 0; k < subspace_dim; k++) {
                subquerypoint[k] = querypoint[j * subspace_dim + k];
            }
            dist += avx2_distance(subquerypoint, codebooks[j][index[i][j]]);
        }
        if (dist < min_dist) {
            min_dist = dist;
            best_idx = i;
        }
    }
    return best_idx;
}

PQ_Codebooks PQ::train(const SiftData<float> &data) {
    int dim = data.get_dim();
    int subspacedim = dim / subspace_num;

    for (int i = 0; i < subspace_num; i++) {
        std::vector<std::vector<float>> subdata(data.get_num(), std::vector<float>(subspacedim));
        for (int j = 0; j < data.get_num(); j++) {
            for (int k = 0; k < subspacedim; k++) {
                subdata[j][k] = data.data[j][i * subspacedim + k];
            }
        }
        long long freq, head, tail;
        QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
        QueryPerformanceCounter((LARGE_INTEGER*)&head);
        KMEANS kmeans(centroid_num, 1);
        codebooks[i] = kmeans.fit_openmp(subdata, 20);
        QueryPerformanceCounter((LARGE_INTEGER*)&tail);
        std::cout << "subspace " << i << " trained time " << (tail - head) * 1000.0 / freq << "ms" << std::endl;
        std::cout << "subspace " << i << " trained. " << "codebooks: (" << codebooks[i].size() << ", " << codebooks[i][0].size() << ")" << std::endl;
    }

    std::cout << "PQ trained. " << "codebooks: (" << codebooks.size() << ", " << codebooks[0].size() << ", " << codebooks[0][0].size() << ")" << std::endl;
    std::cout << "======================================" << std::endl;
    return this->codebooks;
}

void PQ::save_codebooks(const std::string &filename) {
    std::ofstream out(filename, std::ios::binary | std::ios::trunc);
    if (!out.is_open()) {
        std::cerr << "Failed to open file for writing: " << filename << std::endl;
        return;
    }
    int dimensions[3] = {static_cast<int>(codebooks.size()),
                         static_cast<int>(codebooks.empty() ? 0 : codebooks[0].size()),
                         static_cast<int>(codebooks.empty() || codebooks[0].empty() ? 0 : codebooks[0][0].size())};
    out.write(reinterpret_cast<char*>(dimensions), sizeof(dimensions));

    for (const auto& matrix : codebooks) {
        for (const auto& row : matrix) {
            out.write(reinterpret_cast<const char*>(row.data()), row.size() * sizeof(float));
        }
    }

    out.close();

    std::cout << "PQ saved. " << "codebooks: (" << codebooks.size() << ", " << codebooks[0].size() << ", " << codebooks[0][0].size() << ")" << std::endl;
}

void PQ::read_codebooks(const std::string &filename) {
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

    codebooks.resize(outerSize);
    for (int i = 0; i < outerSize; ++i) {
        codebooks[i].resize(middleSize);
        for (int j = 0; j < middleSize; ++j) {
            codebooks[i][j].resize(innerSize);
            in.read(reinterpret_cast<char*>(codebooks[i][j].data()), innerSize * sizeof(float));
            if (in.gcount() != innerSize * sizeof(float)) {
                std::cerr << "Failed to read data properly from file: " << filename << std::endl;
                return;
            }
        }
    }

    in.close();

    std::cout << "PQ read. " << "codebooks: (" << codebooks.size() << ", " << codebooks[0].size() << ", " << codebooks[0][0].size() << ")" << std::endl;
}

PQ_Index PQ::buildIndex(const SiftData<float> &data) {
    std::vector<std::vector<int>> index(data.get_num(), std::vector<int>(subspace_num));
    long long freq, head, tail;
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    for (int i = 0; i < data.get_num(); i++) {
        index[i] = quantize_openmp(data.data[i]);
        if (i % 100000 == 0)
            std::cout << "quantized " << i << " datapoint." << std::endl;
    }
    this->index = index;
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    std::cout << "Indexing time: " << (tail - head) * 1000.0 / freq << "ms" << std::endl;
    std::cout << "Index built. " << "index: (" << index.size() << ", " << index[0].size() << ")" << std::endl;

    return this->index;
}

PQ_Index PQ::buildIndexMPI(const SiftData<float>& data)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int num_vectors = data.get_num();
    int local_size = num_vectors / size;
    int start_index = rank * local_size;
    int end_index = (rank == size - 1) ? num_vectors : start_index + local_size;  // 处理最后一个进程的边界情况

    std::cout << "Process " << rank << " is processing from index " << start_index << " to " << end_index << std::endl;

    PQ_Index local_index(local_size, std::vector<int>(subspace_num));
    for (int i = start_index; i < end_index; ++i) {
        local_index[i - start_index] = quantize_openmp(data.data[i]);
        if ((i - start_index) % 10000 == 0) {
            std::cout << "Process " << rank << " processed " << i - start_index << " items." << std::endl;
        }
    }

    std::vector<std::vector<int>> all_indices;
    if (rank == 0) {
        all_indices.resize(num_vectors, std::vector<int>(subspace_num));
    }
    if (rank == 0) {
        std::cout << "Process " << rank << " is gathering results from all processes." << std::endl;
    }
    //MPI_Gather(local_index.data(), local_size * subspace_num, MPI_INT,
               //all_indices.data(), local_size * subspace_num, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        std::cout << "All processes have completed their work and results have been gathered." << std::endl;
        this->index = all_indices;
    }

    return (rank == 0) ? this->index : PQ_Index(); // 只有主进程返回完整索引
}

void PQ::save_index(const std::string &filename) {
    std::ofstream outFile(filename, std::ios::binary | std::ios::trunc);
    if (!outFile) {
        std::cerr << "Error opening file for writing: " << filename << std::endl;
        return;
    }

    // 获取行数和列数
    size_t rows = index.size();
    size_t cols = rows ? index[0].size() : 0;

    // 写入行数和列数
    outFile.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
    outFile.write(reinterpret_cast<const char*>(&cols), sizeof(cols));

    // 写入矩阵数据
    for (const auto& row : index) {
        outFile.write(reinterpret_cast<const char*>(row.data()), row.size() * sizeof(int));
    }

    outFile.close();

    std::cout << "Index saved. " << "index: (" << index.size() << ", " << index[0].size() << ")" << std::endl;
}

void PQ::read_index(const std::string &filename) {
    std::ifstream inFile(filename, std::ios::binary);
    if (!inFile) {
        std::cerr << "Error opening file for reading: " << filename << std::endl;
        return;
    }

    size_t rows, cols;
    // 读取行数和列数
    inFile.read(reinterpret_cast<char*>(&rows), sizeof(rows));
    inFile.read(reinterpret_cast<char*>(&cols), sizeof(cols));

    index.resize(rows);  // 调整外层向量的大小以容纳所有行
    for (auto& row : index) {
        row.resize(cols);  // 调整每一行的大小
        inFile.read(reinterpret_cast<char*>(row.data()), cols * sizeof(int));  // 直接读取整行数据到向量中
    }

    inFile.close();

    std::cout << "Index read. " << "index: (" << index.size() << ", " << index[0].size() << ")" << std::endl;
}

struct query_ThreadData {
    const SiftData<float>* all_data;
    PQ* pq_instance;
    std::vector<int>* result;
    size_t start_index;
    size_t end_index;
};

void* thread_query(void* arg) {
    query_ThreadData* data = static_cast<query_ThreadData*>(arg);
    for (size_t i = data->start_index; i < data->end_index; i++) {
        (*data->result)[i] = data->pq_instance->asymmetric_query(data->all_data->data[i]);
    }
    return nullptr;
}

std::vector<int> PQ::query_thread(const SiftData<float> &querydata, int thread_num) {
    size_t num_threads = thread_num;  // Or any other appropriate number
    std::vector<int> result(querydata.get_num());
    pthread_t threads[num_threads];
    query_ThreadData threadData[num_threads];

    size_t part = querydata.get_num() / num_threads;
    for (size_t i = 0; i < num_threads; i++) {
        threadData[i].pq_instance = this;
        threadData[i].all_data = &querydata;
        threadData[i].result = &result;
        threadData[i].start_index = i * part;
        threadData[i].end_index = (i + 1) == num_threads ? querydata.get_num() : (i + 1) * part;
        pthread_create(&threads[i], nullptr, thread_query, &threadData[i]);
    }
    for (auto &thread : threads) {
        pthread_join(thread, nullptr);
    }

    return result;
}

std::vector<int> PQ::query_openmp(const SiftData<float> &querydata, int thread_num) {

    std::vector<int> result(querydata.get_num());

    omp_set_num_threads(thread_num);

    #pragma omp parallel for
    for (size_t i = 0; i < querydata.get_num(); i++) {
        result[i] = asymmetric_query(querydata.data[i]);
    }

    return result;
}

std::vector<int> PQ::query_mpi(const SiftData<float> &querydata) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    long long freq, head, tail;
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    QueryPerformanceCounter((LARGE_INTEGER*)&head);

    std::vector<int> local_result;
    int local_size = querydata.get_num() / size;
    int remainder = querydata.get_num() % size;

    int local_start = rank * local_size;
    int local_end = (rank == size - 1) ? querydata.get_num() : (rank + 1) * local_size;

    for (int i = local_start; i < local_end; i++) {
        local_result.push_back(asymmetric_query(querydata.data[i]));
        if ((i - local_start) % 1000 == 0)
            std::cout << "Rank " << rank << " queried " << (i - local_start) << " querypoint." << std::endl;
    }

    std::vector<int> result;
    if (rank == 0) {
        result.resize(querydata.get_num());
    }

    MPI_Gather(local_result.data(), local_result.size(), MPI_INT,
               result.data(), local_size, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0 && remainder > 0) {
        for (int i = size * local_size; i < querydata.get_num(); i++) {
            result[i] = asymmetric_query(querydata.data[i]);
            if ((i - size * local_size) % 1000 == 0)
                std::cout << "Rank 0 queried remaining " << (i - size * local_size) << " querypoint." << std::endl;
        }
    }

    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    if (rank == 0) {
        std::cout << "Query time: " << (tail - head) * 1000.0 / freq << "ms" << std::endl;
    }

    return result;
}

std::vector<int> PQ::query(const SiftData<float> &querydata) {
    long long freq, head, tail;
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    std::vector<int> result;
    for (size_t i = 0; i < querydata.get_num(); i++) {
        result.push_back(asymmetric_query(querydata.data[i]));
//        result.push_back(symmetric_query(querydata.data[i]));
        if (i % 1000 == 0)
            std::cout << "queried " << i << " querypoint." << std::endl;
    }
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    std::cout << "Query time: " << (tail - head) * 1000.0 / freq << "ms" << std::endl;
    return result;
}

double PQ::calc_recall(const std::vector<int> result, const SiftData<int> &groundtruth) {
    double recall = 0.0;
    double count = 0;
    for (int i = 0; i < result.size(); i++) {
        for (int j = 0; j < groundtruth.data[i].size(); j++) {
            if (result[i] == groundtruth.data[i][j]) {
                count++;
                break;
            }
        }
    }
    recall = count / result.size();
    return recall;
}

int PQ::symmetric_query(const std::vector<float> &querypoint) {
    return 0;
}


struct buildIndex_ThreadData {
    const SiftData<float>* all_data;
    PQ* pq_instance;
    unsigned int start_index;
    unsigned int end_index;
    PQ_Index* partial_index;
};

void* quantize_data(void* arg) {
    buildIndex_ThreadData* data = static_cast<buildIndex_ThreadData*>(arg);

    std::vector<std::vector<int>>& part_index = *(data->partial_index);
    for (unsigned int i = data->start_index; i < data->end_index; ++i) {
        part_index[i] = data->pq_instance->quantize_pthread(data->all_data->data[i]);
    }
    pthread_exit(NULL);
}

PQ_Index PQ::buildIndex_pthread(const SiftData<float>& data, int thread_num) {
    unsigned int num_threads = thread_num;  // 可调整线程数量
    unsigned int num_vectors = data.get_num();
    unsigned int vectors_per_thread = num_vectors / num_threads;

    pthread_t threads[num_threads];
    buildIndex_ThreadData thread_data[num_threads];
    PQ_Index index(num_vectors, std::vector<int>());  // 初始化最终的索引

    for (unsigned int i = 0; i < num_threads; ++i) {
        thread_data[i].all_data = &data;
        thread_data[i].pq_instance = this;
        thread_data[i].start_index = i * vectors_per_thread;
        thread_data[i].end_index = (i == num_threads - 1) ? num_vectors : (i + 1) * vectors_per_thread;
        thread_data[i].partial_index = &index;

        pthread_create(&threads[i], NULL, quantize_data, (void*)&thread_data[i]);
    }

    for (unsigned int i = 0; i < num_threads; ++i) {
        pthread_join(threads[i], NULL);
    }
    this->index = index;
    return index;
}

PQ_Index PQ::buildIndex_openmp(const SiftData<float>& data, int thread_num) {
    int num_vectors = data.get_num();

    // 设置线程数
    omp_set_num_threads(thread_num);  // 你可以根据需要调整这个数字

    // OpenMP 并行循环
    #pragma omp parallel for
    for (int i = 0; i < num_vectors; i++) {
        this->index[i] = quantize_openmp(data.data[i]);
    }

    return index;
}
