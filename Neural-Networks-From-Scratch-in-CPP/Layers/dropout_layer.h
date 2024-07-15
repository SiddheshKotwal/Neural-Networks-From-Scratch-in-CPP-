#ifndef dropout_layer
#define dropout_layer

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

// Dropout:
// Dropout is a regularization technique used in neural networks to prevent overfitting. During training, it randomly sets a fraction of input units to 0 at each update,
// which helps prevent the network from becoming too dependent on specific neurons. This forces the network to learn more robust features and improves generalization.
// Dropout is only applied during training and is typically not used during evaluation or testing. The dropout rate, often set between 0.2 and 0.5, determines the fraction
// of neurons to drop. In pytorch it determines the fraction of neurons to keep.

// During drop out only fraction of neurons are used so that decreases the overall impact of data on neurons so the active neurons are scaled by the inverse of dropout rate (i.e. divided by 1 - dropout rate) 
// to accomodate the effect of remaining data on the neurons that are active.
// But this changes the overall sum of the neurons output prior dropout and after dropout but by enough samples the scaling averages out overall, 
// and the prior sum looks similar to after dropout sum of output of neurons.
// Dropout is only used during the training process and not during prediction phase.

class Layer_Dropout{
    
    public:
    vector<vector<double>> output, inputs, binary_mask, dinputs;
    double rate;
    
    Layer_Dropout(double rate){
        // Store rate, we invert it as for example for dropout of 0.1 we need success rate of 0.9
        this->rate = 1 - rate;
    }

    void forward(vector<vector<double>>& inputs){

        // Save input values
        this->inputs = inputs;

        mt19937 generate(0); // Mersenne Twister random number generator
        bernoulli_distribution distribution(this->rate); // (probability of success)

        // Generate and save scaled mask
        this->binary_mask.resize(inputs.size(), vector<double>(inputs[0].size(), 0));
        this->output = binary_mask; // setting same size
        for(int i = 0; i < inputs.size(); i++){
            for(int j = 0; j < inputs[0].size(); j++)
                binary_mask[i][j] = distribution(generate) / this->rate;
        }
        
        // Apply mask to output values
        for(int i = 0; i < inputs.size(); i++){
            for(int j = 0; j < inputs[0].size(); j++)
                this->output[i][j] = inputs[i][j] * this->binary_mask[i][j];
        }
    }

    void backward(vector<vector<double>>& dvalues){
        
        this->dinputs = dvalues; // Setting same size
        // Gradient on values
        for(int i = 0; i < dvalues.size(); i++){
            for(int j = 0; j < dvalues[0].size(); j++)
                this->dinputs[i][j] = dvalues[i][j] * this->binary_mask[i][j];
        }
    }
};

#endif