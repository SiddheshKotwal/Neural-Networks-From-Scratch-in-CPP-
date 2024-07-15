#include "common_includes.h"

// Use y_true label as just a vector not matrix because of the functions defined for vector not matrix of y_true label


// Printing Matrix
void print(vector<vector<double>>& matrix) {
    for (auto i : matrix) {
        for (auto j : i) cout << j << " ";
        cout << "\n";
    }
    cout << "\n";
}

void print(vector<vector<double>>& matrix, string str) {
    cout << str << ":\n";
    for (auto i : matrix) {
        for (auto j : i) cout << j << " ";
        cout << "\n";
    }
    cout << "\n";
}

// Function overloaded for printing vector
void print(vector<double>& vec) {
    for (auto i : vec) cout << i << " ";
    cout << "\n";
}

int main(){

    // 100 samples for each class -> 300 samples 
    long long samples = 100, classes = 3;
    vector<vector<double>> X(samples * classes, vector<double>(2, 0)); // 2 columns(features)
    vector<double> y(samples * classes, 0); // labels

    // generating spiral patterned dataset
    spiral_data(X, y, samples, classes);
    
    // ReLU Activation function object creation
    Activation_ReLU activation1;

    // Output Layer Activation function (Softmax)
    Activation_Softmax activation2;

    // Create Dense layer with 2 input features and 3 output values(number of neurons), 
    // dense2 will be created by checking prev layers no. of outputs as inputs for this layer, and it's no. of neurons 
    // weights for dense2 dim -> (no. of inputs x no. of neurons)
    Layer_Dense dense1(2, 3), dense2(3, 3);

    // Perform a forward pass of our training data through this layer
    dense1.forward(X);
    // print(dense1.output); // dimensions (300, 3)

    // Forward pass through activation function
    activation1.forward(dense1.output); // ReLU output will be (n_samples, n_neurons)
    // print(activation1.output); // negative values are clipped (set to zero)
    // This ReLU output will be the input for next Layer

    // Input for dense2 
    dense2.forward(activation1.output);

    /*  // Using combined softmax and cross entropy loss function class instead of individuals
        
        activation2.forward(dense2.output); // Softmax
        print(activation2.output); // Final output containing confidence scores for each neurons

        // Create Loss Object
        CategoricalCrossentropyLoss loss_function;
        
        // Calculating loss by using categorical cross entropy
        vector<double> pred_losses = loss_function.forward(activation2.output, y);
        double average_loss = loss_function.calculate(pred_losses); // Calculating mean of the losses

    */
    
    // create an object of softmax and cross entropy loss functions combined
    Activation_Softmax_Loss_CategoricalCrossentropy loss_activation;
    double average_loss = loss_activation.forward(dense2.output, y); // performed softmax activation and categorical cross entropy loss with mean loss output

    // Printing Average Loss Value
    cout<< "Mean Loss: " << average_loss <<"\n";

    // Calculating Accuracy of the neural network
    Accuracy_Categorical accuracy;
    double accuracy_ = accuracy.calculate(loss_activation.output, y);
    cout<< "Accuracy: " << accuracy_ << "\n" ;

    // Backward pass
    loss_activation.backward(loss_activation.output, y);
    dense2.backward(loss_activation.dinputs);
    activation1.backward(dense2.dinputs);
    dense1.backward(activation1.dinputs);

    // Print gradients
    print(dense1.dweights, "1dweights");
    print(dense1.dbiases, "1dbiases");
    print(dense2.dweights, "2dweights");
    print(dense2.dbiases, "2dbiases");

    Optimizer_SGD optimizer;
    optimizer.update_params(dense1);
    optimizer.update_params(dense2);

    return 0;
}