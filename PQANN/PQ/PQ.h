//
// Created by Kaiser Tan on 2024/4/21.
//

#ifndef PQANN_PQ_H
#define PQANN_PQ_H

#include <vector>
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

    PQ_Index buildIndex(const SiftData<float>& data);

    int asymmetric_query(const std::vector<float>& querypoint);

    int symmetric_query(const std::vector<float>& querypoint);

    std::vector<int> query(const SiftData<float>& querydata);

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

private:
    PQ_Codebooks codebooks;
    PQ_Index index;
    unsigned int subspace_num;
    unsigned int centroid_num;
};


#endif //PQANN_PQ_H
