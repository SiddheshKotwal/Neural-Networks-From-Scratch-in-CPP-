#include "common_includes.h"

// Applying Random weights and finding the minimum loss and maximum accuracy to get better fit for neural network
// First applying directly random weights and biases for all layers neurons we are barely able to improve the performance
// Then we tried by updating the best weights and best biases by adding fraction of random values and if loss is decreased we change our 
// best parameters else we copy prev best parameters in current parameters.
// By this method loss decreased significantly and also accuracy increased
// But when tried with complex dataset i.e spiral dataset it was barely able to find parameters that decreased loss and also accuracy was
// not increased reasonably due to the complexity of the dataset it may have not able to improve it's because of local minimum of loss.

int main(){

    long long samples = 100, classes = 3;
    vector<vector<double>> X(samples * classes, vector<double>(2, 0));
    vector<double> y(samples * classes, 0);

    mt19937 randm(0);
    normal_distribution<double> distribute(0, 1);

    // Create Dataset
    // vertical_data(X, y, samples, classes);
    spiral_data(X, y, samples, classes);

    // Create Model
    Layer_Dense dense1(2, 3), dense2(3, 3);
    Activation_ReLU activation1;
    Activation_Softmax activation2;

    // Create Loss Function
    CategoricalCrossentropyLoss loss_funcions;
    Accuracy_Categorical accuracy;

    // Helper Variables
    double lowest_loss = 9999999; // some initial value
    vector<vector<double>> best_dense1_weights = dense1.weights;
    vector<vector<double>> best_dense1_biases = dense1.biases;
    vector<vector<double>> best_dense2_weights = dense2.weights;
    vector<vector<double>> best_dense2_biases = dense2.biases;

    for(int i = 0; i < 10000; i++){

        // Generate a new set of weights for iteration
        for(int i = 0; i < dense1.weights.size(); i++){
            for(int j = 0; j < dense1.weights[0].size(); j++)
                dense1.weights[i][j] += distribute(randm) * 0.05;
        }

        for(int i = 0; i < dense1.biases.size(); i++){
            for(int j = 0; j < dense1.biases[0].size(); j++)
                dense1.biases[i][j] += distribute(randm) * 0.05;
        }

        for(int i = 0; i < dense2.weights.size(); i++){
            for(int j = 0; j < dense2.weights[0].size(); j++)
                dense2.weights[i][j] += distribute(randm) * 0.05;
        }

        for(int i = 0; i < dense2.biases.size(); i++){
            for(int j = 0; j < dense2.biases[0].size(); j++)
                dense2.biases[i][j] += distribute(randm) * 0.05;
        }

        // Perform a forward pass of the training data through this layer
        dense1.forward(X);
        activation1.forward(dense1.output);
        dense2.forward(activation1.output);
        activation2.forward(dense2.output);

        // Perform a forward pass through activation function
        // it takes the output of second dense layer here and returns loss
        vector<double> pred_loss = loss_funcions.forward(activation2.output, y);
        double loss = loss_funcions.calculate(pred_loss);

        // Calculate accuracy from output of activation2 and targets
        // calculate values along first axis
        accuracy.compare(activation2.output, y);
        double accuracy_ = accuracy.calculate();

        // If loss is smaller - print and save weights and biases aside
        if(loss < lowest_loss){
            cout<<"New set of weights found, iteration: " << i << " loss: " << loss << " acc: " << accuracy_ << "\n" ;
            best_dense1_weights = dense1.weights;
            best_dense1_biases = dense1.biases;
            best_dense2_weights = dense2.weights;
            best_dense2_biases = dense2.biases;
            lowest_loss = loss;
        }
        else{
            dense1.weights = best_dense1_weights;
            dense1.biases = best_dense1_biases;
            dense2.weights = best_dense2_weights;
            dense2.biases = best_dense2_biases;
        }
    }
}