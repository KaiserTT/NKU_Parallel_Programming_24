//
// Created by Kaiser Tan on 2024/4/23.
//

#ifndef PQANN_RPQ_H
#define PQANN_RPQ_H

#include <vector>
#include "../SiftData/SiftData.h"
#include "../PQ/PQ.h"

typedef std::vector<std::vector<std::vector<std::vector<float>>>> RPQ_Codebooks;
typedef std::vector<std::vector<std::vector<int>>> RPQ_Index ;

class RPQ {
public:
    RPQ(unsigned int subspace_num, unsigned int centroid_num, unsigned int layers) :
        pq(subspace_num, centroid_num),
        layers(layers),
        codebooks(layers, std::vector<std::vector<std::vector<float>>>(subspace_num, std::vector<std::vector<float>>(centroid_num, std::vector<float>()))),
        index(layers, std::vector<std::vector<int>>()) {}

    ~RPQ() = default;

    void train(const SiftData<float>& siftData);

    SiftData<float> calc_residuals(const SiftData<float>& siftData, const PQ_Index& index, const PQ_Codebooks& codebooks);

    std::vector<float> calc_datapoint_residuals(const std::vector<float>& datapoint0, const std::vector<float>& datapoint1);

    std::vector<float> RPQ::calc_datapoint_residuals_avx2(const std::vector<float> &datapoint0, const std::vector<float> &datapoint1);

    std::vector<float> calc_datapoint_residuals_unroll(const std::vector<float>& datapoint0, const std::vector<float>& datapoint1);

    unsigned int asymmetric_query(const std::vector<float>& querypoint);

    void save_codebooks(const std::string& filename);

    void read_codebooks(const std::string& filename);

    void save_index(const std::string& filename);

    void read_index(const std::string& filename);

private:
    PQ pq;
    unsigned int layers;
    RPQ_Codebooks codebooks;
    RPQ_Index index;
};


#endif //PQANN_RPQ_H
