#ifndef calculate_accuracy
#define calculate_accuracy

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>
#include <omp.h>
using namespace std;

class Accuracy{
    public:
    double accumulated_count, accumulated_sum;
    vector<vector<double>> comparisons_;
    vector<double> comparisons;

    double calculate(){

        double total = 0, size = 0;
        if(comparisons_.size() > 0){
            for(auto i : comparisons_){
                for(auto j : i) if(j == 1) total++;
            }
            size = comparisons_.size() * comparisons_[0].size();
        }
        else{
            for(auto i : comparisons) if(i == 1) total++;
            size = comparisons.size();
        }
        
        this->accumulated_sum += total;
        this->accumulated_count += size;
        return total / size;
    }

    double calculate_accumulated(){
        return this->accumulated_sum / this->accumulated_count;
    }

    void new_pass(){
        this->accumulated_count = this->accumulated_sum = 0;
    }
};

#endif