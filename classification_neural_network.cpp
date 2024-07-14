#include "commonIncludes.h"
#include "datasets/spiral.cpp"
#include "Layers/dropout_layer.cpp"
#include "Layers/dense_layer.cpp"
#include "accuracy/categorical.cpp"
#include "activation_functions/ReLU.cpp"
#include "activation_functions/Softmax.cpp"
#include "loss_functions/categorical_cross_entropy.cpp"
#include "classifier/softmax_classifier.cpp"
#include "optimizers/stochastic_gradient_descent.cpp"

// Classification neural network

int main(){

    long long samples = 100, classes = 3;
    MatrixXd X, X_test;
    VectorXd y, y_test;

    // Create Dataset
    create_data(X, y, samples, classes);

    Layer_Dense dense1(2, 64, 0, 5e-4, 0, 5e-4);
    Activation_ReLU activation1;
    Layer_Dense dense2(64, 3);
    Layer_Dropout dropout1(0.1);
    Accuracy_Categorical accuracy;
    Activation_Softmax_Loss_CategoricalCrossentropy loss_activation;
    Optimizer_SGD optimizer(1.0, 1e-3, 0.9);

    long long epoch = 10001;
    for(int i = 0; i < epoch; i++){
        
        // Forward pass
        dense1.forward(X);
        activation1.forward(dense1.output);
        dropout1.forward(activation1.output);
        dense2.forward(dropout1.output);

        double data_loss = loss_activation.forward(dense2.output, y);
        double reg_loss = loss_activation.loss_function.regularization_loss(dense1) + loss_activation.loss_function.regularization_loss(dense2);
        double loss = data_loss + reg_loss;
        double acc = accuracy.calculate(loss_activation.output, y);

        if(!(i % 100)) cout<<"epoch: "<<i<<", acc: "<<acc<<", loss: "<<loss<<", data_loss: "<<data_loss<<", reg_loss: "<<reg_loss<<", lr: "<<optimizer.current_learning_rate<<"\n";

        // Backward Pass
        loss_activation.backward(loss_activation.output, y);
        dense2.backward(loss_activation.dinputs);
        dropout1.backward(dense2.dinputs);
        activation1.backward(dropout1.dinputs);
        dense1.backward(activation1.dinputs);

        // Update weights and biases
        optimizer.pre_update_params();
        optimizer.update_params(dense1);
        optimizer.update_params(dense2);
        optimizer.post_update_params();
    }
    
    create_data(X_test, y_test, samples, classes); // testing set

    dense1.forward(X_test);
    activation1.forward(dense1.output);
    dense2.forward(activation1.output);
    
    double data_loss = loss_activation.forward(dense2.output, y_test);
    double reg_loss = loss_activation.loss_function.regularization_loss(dense1) + loss_activation.loss_function.regularization_loss(dense2);
    double loss = data_loss + reg_loss;
    double val_accuracy = accuracy.calculate(loss_activation.output, y_test);
    cout<<"Validation, acc: "<<val_accuracy<<", loss: "<<loss<<", data_loss: "<<data_loss<<", reg_loss: "<<reg_loss<<"\n";
    
    return 0;
}
