#include <iostream>
#include <vector>
#include <omp.h>
#include <fstream>
#include <sstream>
#include "datasets/spiral.cpp"
#include "dense_layer.cpp"
#include "helper_functions.cpp"
#include "activation_functions/ReLU.cpp"
#include "activation_functions/Softmax.cpp"
#include "activation_functions/Sigmoid.cpp"
#include "loss_functions/binary_cross_entropy.cpp"
#include "optimizer/stochastic_gradient_descent.cpp"
#include "optimizer/adaptive_gradient.cpp"
#include "optimizer/root_mean_square_propagation.cpp"
#include "optimizer/adaptive_moment_estimation.cpp"
using namespace std;

// Binary Logistic Regression Model

int main(){

    long long samples = 100, classes = 2;
    vector<vector<double>> X, X_test, y_resized, y_test_resized;
    vector<double> y, y_test;
    
    // Create Dataset
    spiral_data(X, y, samples, classes);
    y_resized.resize(y.size(), vector<double>(1, 0));
    for(int i = 0; i < y.size(); i++)
        y_resized[i][0] = y[i];

    Layer_Dense dense1(2, 64, 0, 5e-4, 0, 5e-4);
    Activation_ReLU activation1;
    Layer_Dense dense2(64, 1);
    Activation_Sigmoid activation2;
    Loss_BinaryCrossentropy loss_function;
    Optimizer_Adam optimizer(0.002, 5e-7);

    long long epoch = 10001;
    for(int i = 0; i < epoch; i++){
        
        dense1.forward(X);
        activation1.forward(dense1.output);
        dense2.forward(activation1.output);
        activation2.forward(dense2.output);

        vector<double> sample_loss = loss_function.forward(activation2.output, y_resized);
        double data_loss = loss_function.calculate(sample_loss);
        double reg_loss = loss_function.regularization_loss(dense1) + loss_function.regularization_loss(dense2);
        double loss = data_loss + reg_loss;

        vector<vector<double>> pred;
        double total = 0, accuracy_ = 0;
        pred = activation2.output;
        for(int i = 0; i < pred.size(); i++){
            for(int j = 0; j < pred[0].size(); j++){
                pred[i][j] = (activation2.output[i][j] > 0.5) * 1;
                if(pred[i][j] == y_resized[i][j]) total++;
            }
        }
        accuracy_ = total / (pred.size() * pred[0].size());

        if(!(i % 100)) cout<<"epoch: "<<i<<", acc: "<<accuracy_<<", loss: "<<loss<<", data_loss: "<<data_loss<<", reg_loss: "<<reg_loss<<", lr: "<<optimizer.current_learning_rate<<"\n";

        loss_function.backward(activation2.output, y_resized);
        activation2.backward(loss_function.dinputs);
        dense2.backward(activation2.dinputs);
        activation1.backward(dense2.dinputs);
        dense1.backward(activation1.dinputs);

        // Update weights and biases
        optimizer.pre_update_params();
        optimizer.update_params(dense1);
        optimizer.update_params(dense2);
        optimizer.post_update_params();
    }

    spiral_data(X_test, y_test, samples, classes); // testing set

    y_test_resized.resize(y_test.size(), vector<double>(1, 0));
    for(int i = 0; i < y_test.size(); i++)
        y_test_resized[i][0] = y_test[i];

    dense1.forward(X_test);
    activation1.forward(dense1.output);
    dense2.forward(activation1.output);
    activation2.forward(dense2.output);
    vector<double> sample_loss = loss_function.forward(activation2.output, y_test_resized);

    double data_loss = loss_function.calculate(sample_loss);
    double reg_loss = loss_function.regularization_loss(dense1) + loss_function.regularization_loss(dense2);
    double loss = data_loss + reg_loss;
    
    vector<vector<double>> pred;
    double total = 0, accuracy_;
    pred = activation2.output;
    for(int i = 0; i < pred.size(); i++){
        for(int j = 0; j < pred[0].size(); j++){
            pred[i][j] = (activation2.output[i][j] > 0.5) * 1;
            if(pred[i][j] == y_resized[i][j]) total++;
        }
    }
    accuracy_ = total / (pred.size() * pred[0].size());
    cout<<"Validation, acc: "<<accuracy_<<", loss: "<<loss<<", data_loss: "<<data_loss<<", reg_loss: "<<reg_loss<<"\n";
    
    return 0;
}