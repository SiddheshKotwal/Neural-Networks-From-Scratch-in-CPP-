#ifndef vertical
#define vertical

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

// Generating random numbers from gaussian distribution of mean 0 and variance 1 -> (-1, 1)
mt19937 gen(0); // seed 0
normal_distribution<> d(0, 1); // mean = 0, var = 1

void vertical_data(vector<vector<double>>& X, vector<double>& y, long long samples, int classes){
    X.resize(samples * classes, std::vector<double>(2));
    y.resize(samples * classes);

    for(int i = 0; i < classes; i++){

        long long m = i * samples, n = (i + 1) * samples;
        for(int j = m; j < n; j++){
            X[j][0] = d(gen) * 0.1 + (i / 3.0);
            X[j][1] = d(gen) * 0.1 + 0.5;
            y[j] = i;
        }
    }
}

#endif