#include <iostream>
#include <vector>
#include <tuple>
#include <fstream>
#include "cereal/archives/binary.hpp"
#include <cereal/types/vector.hpp>
#include <cereal/types/tuple.hpp>

template <typename T>
void save_parameters(const T& data, const std::string& filename) {
    std::ofstream ofs(filename, std::ios::binary);
    cereal::BinaryOutputArchive archive(ofs);
    archive(data);
}

template <typename T>
void load_parameters(T& data, const std::string& filename) {
    std::ifstream ifs(filename, std::ios::binary);
    cereal::BinaryInputArchive archive(ifs);
    archive(data);
}
