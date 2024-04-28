//
// Created by Kaiser Tan on 2024/4/21.
//

#ifndef PQANN_SIFTDATA_H
#define PQANN_SIFTDATA_H

#include <vector>
#include <string>
#include <fstream>
#include <iostream>

template <typename T>
class SiftData {
public:
    SiftData() : num(0), dim(0) {}
    SiftData(unsigned int num, unsigned int dim) : num(num), dim(dim), data(num, std::vector<T>(dim)) {}
    ~SiftData() = default;

    void load_vecs_data(const std::string& filename);

    unsigned int get_num() const { return num; }

    unsigned int get_dim() const { return dim; }

    std::vector<std::vector<T>> data;

protected:
    unsigned int num;
    unsigned int dim;
};

template <typename T>
void SiftData<T>::load_vecs_data(const std::string &filename) {
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
        std::cerr << "Cannot open file " << filename << std::endl;
        exit(1);
    }

    in.read((char *) &this->dim, 4);
    in.seekg(0, std::ios::end);
    std::ios::pos_type ss = in.tellg();
    size_t fsize = (size_t) ss;
    this->num = (unsigned int) (fsize / (dim + 1) / 4);
    in.seekg(0, std::ios::beg);

    data.resize(num);
    for (unsigned int i = 0; i < num; i++) { data[i].resize(dim); }
    for (size_t i = 0; i < num; i++) {
        in.seekg(4, std::ios::cur);
        in.read((char *) data[i].data(), dim * 4);
    }

    in.close();
}


#endif //PQANN_SIFTDATA_H
