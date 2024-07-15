#ifndef ReLU
#define ReLU

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

class Activation_ReLU{

    // Without activation function or while using linear Activation function our neural networks output will be always linear
    // and nonlinear functions will be mapped by linear function resulting in incorrect mapping
    // Neural Networks are especially used for NonLinear mapping

    public:
    vector<vector<double>> output, dinputs, inputs;
    
    // ReLU Activation Function less than or equal to zero values are set to zero else no change 
    // ReLU (Rectified Linear Units) is most commonly used Activation function and easy to implement and efficient to map nonlinear functions
    // Bias contributes in offsetting(shifting) the entire function horizontally and weights contributes in slopes and direction
    void forward(vector<vector<double>>& layer_output){

        // Required for backward pass
        this->inputs = layer_output;

        this->output.resize(layer_output.size(), vector<double>(layer_output[0].size(), 0));
        for(int i = 0; i < layer_output.size(); i++){
            for(int j = 0; j < layer_output[0].size(); j++){
                if(layer_output[i][j] < 0) this->output[i][j] = 0;
                else this->output[i][j] = layer_output[i][j];
            }
        }
    }

    // Backward pass
    void backward(vector<vector<double>>& dvalues){

        // Since we need to modify the original variable,
        // let's make a copy of the values first
        this->dinputs = dvalues; // (number of samples, number of neurons in the layer)

        // Zero gradient where input values were negative
        for(int i = 0; i < dvalues.size(); i++){
            for(int j = 0; j < dvalues[0].size(); j++){
                this->dinputs[i][j] = (this->inputs[i][j] > 0) ? this->dinputs[i][j] : 0;
            }
        }
    }
};

#endif