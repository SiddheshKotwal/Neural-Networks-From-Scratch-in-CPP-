#include "common_includes.h"

// Classification neural network

int main(){

    long long samples = 100, classes = 3;
    vector<vector<double>> X, X_test;
    vector<double> y, y_test;

    // Create Dataset
    spiral_data(X, y, samples, classes);

    // Create Dense layer with 2 input features and 64 output values
    Layer_Dense dense1(2, 64, 0, 5e-4, 0, 5e-4);  // (n_inputs, n_neurons, lambdaL1, lambdaL2)
    // We usually add regularization terms to the hidden layers only. Even if we are calling the
    // regularization method on the output layer as well, it won’t modify gradients if we do not set the
    // lambda hyperparameters to values other than 0.
    // The strength of regularizer increases as we increase the lamda regularizer

    // In theory, this regularization allows us to create much larger models without fear of overfitting (or memorization)
    // We could increase the number of neurons/layers to make larger model still due to regularization the overfitting is prevented.
    // And the as Training data grows, the delta(difference) b/w train_loss/acc and test_loss/acc also decreases meaning no overfitting.

    // Create ReLU activation (to be used with Dense layer)
    Activation_ReLU activation1;

    // Create second Dense layer with 64 input features (as we take output
    // of previous layer here) and 3 output values (output values)
    Layer_Dense dense2(64, 3);

    // Create a dropout layer
    Layer_Dropout dropout1(0.1);
    
    // Create Softmax classifier's combined loss and activation
    Activation_Softmax_Loss_CategoricalCrossentropy loss_activation;
    // Create optimizer
    // Optimizer_SGD optimizer(1.0, 1e-3, 0.9);
    // Optimizer_Adagrad optimizer(1.0, 1e-4);
    // Optimizer_RMSprop optimizer(0.02, 1e-5);
    Optimizer_Adam optimizer(0.005, 5e-7);
    Accuracy_Categorical accuracy;

    // Each full pass through all of the training data is called an epoch
    long long epoch = 10001;
    for(int i = 0; i < epoch; i++){
        
        // Perform a forward pass of our training data through this layer
        dense1.forward(X);

        // Perform a forward pass through activation function
        // takes the output of first dense layer here
        activation1.forward(dense1.output);
        dropout1.forward(activation1.output); // Perform a forward pass through Dropout layer

        // Perform a forward pass through second Dense layer
        // takes outputs of activation function of first layer as inputs
        dense2.forward(dropout1.output);

        // Perform a forward pass through the activation/loss function
        // takes the output of second dense layer here and returns loss
        double data_loss = loss_activation.forward(dense2.output, y);
        double reg_loss = loss_activation.loss_function.regularization_loss(dense1) + loss_activation.loss_function.regularization_loss(dense2);
        double loss = data_loss + reg_loss;

        // Calculate accuracy from output of activation2 and targets
        accuracy.compare(loss_activation.output, y);
        double accuracy_ = accuracy.calculate();

        if(!(i % 100)) cout<<"epoch: "<<i<<", acc: "<<accuracy_<<", loss: "<<loss<<", data_loss: "<<data_loss<<", reg_loss: "<<reg_loss<<", lr: "<<optimizer.current_learning_rate<<"\n";
        // lower loss is not always associated with higher accuracy
        // Due to the Regularization tactics (L1, L2 and dropout)  usually the validation accuracy is 
        // greater than training accuracy because we are not applying dropout layer for predictions, and if it's not then that might be a sign of overfitting.

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

    // Testing dataset to check for overfitting the training acc and loss should be close if not equal for both testing and training datsets, 
    // which would tell us that our model is not overfitted to the training dataset.

    // Some causes of overfitting might be, the learning rate is too high, there are too many training epochs,
    // or the model is too big

    // In general, the goal is to have the testing loss identical to the training loss,
    // even if that means higher loss and lower accuracy on the training data. Similar performance on
    // both datasets means that model generalized instead of overfitting on the training data.

    // If the model is not learning at all we should try larger model, but if the model is learning , but there’s a
    // divergence between the training and testing data it could mean that you should try a smaller model

    // One general rule to follow when selecting initial model hyperparameters is to find the smallest model possible that still learns.
    // Try different hyperparameter settings and select the best one. The reasoning here is that the fewer neurons you have, the less chance
    // you have that the model is memorizing the data. Small number of neurons Fewer neurons can mean it’s easier for a neural
    // network to generalize (actually learn the meaning of the data) compared to memorizing the data.

    spiral_data(X_test, y_test, samples, classes); // testing set

    dense1.forward(X_test);
    activation1.forward(dense1.output);
    dense2.forward(activation1.output);
    
    double data_loss = loss_activation.forward(dense2.output, y_test);
    double reg_loss = loss_activation.loss_function.regularization_loss(dense1) + loss_activation.loss_function.regularization_loss(dense2);
    double loss = data_loss + reg_loss;
    accuracy.compare(loss_activation.output, y_test);
    double val_accuracy = accuracy.calculate();
    cout<<"Validation, acc: "<<val_accuracy<<", loss: "<<loss<<", data_loss: "<<data_loss<<", reg_loss: "<<reg_loss<<"\n";
    
    return 0;

    // Dataset distribution:
    // Training, Validation, Testing
    // Training to train the data with certain hyperparameter settings and then validate on the validation set then check the results
    // Then again try different hyperparameter settings and validate on validation set certain number of times and finally select the best performing
    // hyperparameters and test for final accuracy on Test set which has never been seen by the model. When we have less data we can use k-fold cross validation.

    // Data Preprocessing:
    // For neural networks we need to scale the datasets into range 0 to 1 or -1 to 1. And this scaling has to be done on each of the set training, testing and validation set.
    // We scale these values into these ranges because these values are not exploding due to multiplications and bias additions which can cause integer overflows
    // And also our neural network functions like sigmoid, softmax work properly within range 0 to 1 and tanh within range -1 to 1.

    // Data Augmentation:
    // Data augmentation is a technique we can use to generate data from existing data like we can change the brightness of the images, rotate the images,
    // crop the images and slightly modfiy images untill and unless they can occur in reality. In general, if we use augmentation, then it’s only useful if the augmentations that
    // we make are similar to variations that we could see in reality.

    // Proper size of the dataset for particular problem depends upon the features we want to capture from the dataset and the number of classes we want to predict but there's no single answer for all problems.
    // For more number of features we could require thousands of samples, for only 2-3 features hundreds of samples might be enough.
}
