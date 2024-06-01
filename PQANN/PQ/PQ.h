//
// Created by Kaiser Tan on 2024/4/21.
//

#ifndef PQANN_PQ_H
#define PQANN_PQ_H

#include <vector>
#include <omp.h>
#include "../SiftData/SiftData.h"

typedef std::vector<std::vector<std::vector<float>>> PQ_Codebooks;
typedef std::vector<std::vector<int>> PQ_Index;

class PQ {
public:
    PQ(unsigned int subspace_num, unsigned int centroid_num) :
    subspace_num(subspace_num),
    centroid_num(centroid_num),
    codebooks(subspace_num, std::vector<std::vector<float>>(centroid_num, std::vector<float>())),
    index(subspace_num, std::vector<int>())
    {}

    ~PQ() = default;

    PQ_Codebooks train(const SiftData<float>& data);

    std::vector<int> quantize(const std::vector<float>& datapoint);

    std::vector<int> quantize_pthread(const std::vector<float>& datapoint);

    std::vector<int> quantize_openmp(const std::vector<float>& datapoint);

    PQ_Index buildIndex(const SiftData<float>& data);

    PQ_Index buildIndex_pthread(const SiftData<float>& data, int thread_num);

    PQ_Index buildIndex_openmp(const SiftData<float>& data, int thread_num);

    int asymmetric_query(const std::vector<float>& querypoint);

    int symmetric_query(const std::vector<float>& querypoint);

    std::vector<int> query(const SiftData<float>& querydata);

    std::vector<int> query_openmp(const SiftData<float> &querydata, int thread_num);

    std::vector<int> query_thread(const SiftData<float> &querydata, int thread_num);

    void save_codebooks(const std::string& filename);

    void read_codebooks(const std::string& filename);

    void save_index(const std::string& filename);

    void read_index(const std::string& filename);

    double calc_recall(const std::vector<int> result, const SiftData<int>& groundtruth);

    unsigned int get_subspace_num() const {
        return subspace_num;
    }

    unsigned int get_centroid_num() const {
        return centroid_num;
    }

    PQ_Codebooks get_codebooks() const {
        return codebooks;
    }

    PQ_Index get_index() const {
        return index;
    }

private:
    PQ_Codebooks codebooks;
    PQ_Index index;
    unsigned int subspace_num;
    unsigned int centroid_num;
};


#endif //PQANN_PQ_H
