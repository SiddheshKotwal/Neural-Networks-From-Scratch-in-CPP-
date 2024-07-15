#ifndef Linear
#define Linear

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

class Activation_Linear{
    public:
    vector<vector<double>> inputs, output, dinputs;

    // Forward pass
    void forward(vector<vector<double>> inputs){
        // Just remember values
        this->inputs = inputs;
        this->output = inputs;
    }

    // Backward pass
    void backward(vector<vector<double>> dvalues){
        // derivative is 1, 1 * dvalues = dvalues - the chain rule
        this->dinputs = dvalues;
    }
};

#endif