#include "common_includes.h"

// Commands to run this code:
// g++ regression_model.cpp -I "C:\Program Files\WindowsApps\PythonSoftwareFoundation.Python.3.9_3.9.3568.0_x64__qbz5n2kfra8p0\Include" -I "c:\users\lenovo\appdata\local\packages\pythonsoftwarefoundation.python.3.9_qbz5n2kfra8p0\localcache\local-packages\python39\site-packages\numpy\core\include" -L "C:\Program Files\WindowsApps\PythonSoftwareFoundation.Python.3.9_3.9.3568.0_x64__qbz5n2kfra8p0\libs" -I "C:\mingw64\mingw64\bin" -lPython39
// or ctrl + shift + B
// ./run_my_app

vector<double> get_vector(vector<vector<double>>& vec) {
    vector<double> result(vec.size(), 0);
    for (int i = 0; i < vec.size(); i++)
        result[i] = vec[i][0];
    return result;
}

int main(){

    vector<vector<double>> X, y, Xtest, ytest;
    sine_data(X, y);

    Layer_Dense dense1(1, 64);
    Activation_ReLU activation1;
    Layer_Dense dense2(64, 64);
    Activation_ReLU activation2;
    Layer_Dense dense3(64, 1);
    Activation_Linear activation3;
    Loss_MeanSquaredError loss_function;
    Optimizer_Adam optimizer(0.002, 1e-3);   // (0.002, 1e-3) -> 90%+ accuracy
    Accuracy_Regression accuracy;
    
    // jumping back and forth in accuracy during neural network training can indicate that the learning rate might be too high. 
    // A high learning rate can cause the model to make large updates to the weights, leading to instability in the training process.
    // This instability can manifest as fluctuations in the accuracy, where the model overshoots the optimal point in the loss landscape, 
    // causing the accuracy to oscillate rather than converge smoothly.

    // Accuracy precision for accuracy calculation
    // There are no really accuracy factor for regression problem,
    // but we can simulate/approximate it. We'll calculate it by checking
    // how many values have a difference to their ground truth equivalent
    // less than given precision
    // We'll calculate this precision as a fraction of standard deviation
    // of al the ground truth values

    long long epoch = 10001;
    for(int i = 0; i < epoch; i++){

        // Forward pass
        dense1.forward(X);
        activation1.forward(dense1.output);
        dense2.forward(activation1.output);
        activation2.forward(dense2.output);
        dense3.forward(activation2.output);
        activation3.forward(dense3.output);

        vector<double> sample_losses = loss_function.forward(activation3.output, y);
        double data_loss = loss_function.calculate(sample_losses);
        double reg_loss = loss_function.regularization_loss(dense1) + loss_function.regularization_loss(dense2) + loss_function.regularization_loss(dense3);
        double loss = data_loss + reg_loss;

        // Calculate accuracy from output of activation2 and targets
        // To calculate it we're taking absolute difference between
        // predictions and ground truth values and compare if differences
        // are lower than given precision value
        accuracy.compare(activation3.output, y);
        double acc = accuracy.calculate();
        if(!(i % 100)) cout<<"epoch: "<<i<<", acc: "<<acc<<", loss: "<<loss<<", data_loss: "<<data_loss<<", reg_loss: "<<reg_loss<<", lr: "<<optimizer.current_learning_rate<<"\n";

        // Backward pass
        loss_function.backward(activation3.output, y);
        activation3.backward(loss_function.dinputs);
        dense3.backward(activation3.dinputs);
        activation2.backward(dense3.dinputs);
        dense2.backward(activation2.dinputs);
        activation1.backward(dense2.dinputs);
        dense1.backward(activation1.dinputs);

        // Update weights and biases
        optimizer.pre_update_params();
        optimizer.update_params(dense1);
        optimizer.update_params(dense2);
        optimizer.update_params(dense3);
        optimizer.post_update_params();
    }

    sine_data(Xtest, ytest);

    dense1.forward(Xtest);
    activation1.forward(dense1.output);
    dense2.forward(activation1.output);
    activation2.forward(dense2.output);
    dense3.forward(activation2.output);
    activation3.forward(dense3.output);
    vector<double> X_test, y_test, activation3_output;
    X_test = get_vector(Xtest);
    y_test = get_vector(ytest);
    activation3_output = get_vector(activation3.output);
    plt::plot(X_test, y_test);
    plt::plot(X_test, activation3_output);
    plt::show();
}