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
            double dist = distance_unroll(subdatapoint, this->codebooks[i][j]);
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
            dist += distance_unroll(subquerypoint, codebooks[j][index[i][j]]);
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
        KMEANS kmeans(centroid_num, 200);
        codebooks[i] = kmeans.fit(subdata);
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
        index[i] = quantize(data.data[i]);
        if (i % 100000 == 0)
            std::cout << "quantized " << i << " datapoint." << std::endl;
    }
    this->index = index;
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    std::cout << "Indexing time: " << (tail - head) * 1000.0 / freq << "ms" << std::endl;
    std::cout << "Index built. " << "index: (" << index.size() << ", " << index[0].size() << ")" << std::endl;

    return this->index;
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




