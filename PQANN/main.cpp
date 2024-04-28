#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include "SiftData/SiftData.h"
#include "Kmeans/KMEANS.h"
#include "PQ/PQ.h"
#include "PQ/RPQ.h"
#include "calc/calc.h"
#include <limits>

#include <windows.h>

void load_data(SiftData<float> &siftBase, SiftData<float> &siftQuery, SiftData<int> &siftGroundTruth, SiftData<float> &siftLearn) {
    siftBase.load_vecs_data("../sift/sift_base.fvecs");
    std::cout << "SiftData: sift_base: " << "num = " << siftBase.get_num() << ", dim = " << siftBase.get_dim() << std::endl;

    siftQuery.load_vecs_data("../sift/sift_query.fvecs");
    std::cout << "SiftData: sift_query: " << "num = " << siftQuery.get_num() << ", dim = " << siftQuery.get_dim() << std::endl;


    siftGroundTruth.load_vecs_data("../sift/sift_groundtruth.ivecs");
    std::cout << "SiftData: sift_groundtruth: " << "num = " << siftGroundTruth.get_num() << ", dim = " << siftGroundTruth.get_dim() <<std::endl;


    siftLearn.load_vecs_data("../sift/sift_learn.fvecs");
    std::cout << "SiftData: sift_learn: " << "num = " << siftLearn.get_num() << ", dim = " << siftLearn.get_dim() << std::endl;
    std::cout << "======================================" << std::endl;
}

int main() {
    SiftData<float> siftBase;
    SiftData<float> siftQuery;
    SiftData<int> siftGroundTruth;
    SiftData<float> siftLearn;

    load_data(siftBase, siftQuery, siftGroundTruth, siftLearn);


    return 0;
}
