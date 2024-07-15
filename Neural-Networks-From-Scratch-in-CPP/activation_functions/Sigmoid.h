#ifndef Sigmoid
#define Sigmoid

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

class Activation_Sigmoid{
    public:
    vector<vector<double>> inputs, output, dinputs;

    void forward(vector<vector<double>>& inputs){

        // Save input and calculate/save output
        // of the sigmoid function
        this->inputs = inputs;
        this->output.resize(inputs.size(), vector<double>(inputs[0].size(), 0));
        for(int i = 0; i < inputs.size(); i++){
            for(int j = 0; j < inputs[0].size(); j++)
                this->output[i][j] = 1 / (1 + exp(-1 * inputs[i][j]));
        }
    }

    void backward(vector<vector<double>>& dvalues){

        // Gradient of sigmoid function is sig(i, j) * (1 - sig(i, j))
        this->dinputs.resize(this->output.size(), vector<double>(this->output[0].size(), 0));
        for(int i = 0; i < dinputs.size(); i++){
            for(int j = 0; j < dinputs[0].size(); j++)
                this->dinputs[i][j] = dvalues[i][j] * (1 - this->output[i][j]) * this->output[i][j];
        }
    }
};

#endif