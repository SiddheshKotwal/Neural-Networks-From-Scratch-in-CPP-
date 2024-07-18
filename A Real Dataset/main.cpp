#include "F:\Codes\Neural Networks From Scratch\Github Repo\Neural-Networks-From-Scratch-in-CPP\common_includes.h"
#include <opencv2/opencv.hpp>
#include "get_dataset.cpp"

// Ensure that both training and testing data are scaled using identical methods. 
// Preprocessing rules should be derived solely from the training dataset. Any preprocessing rules 
// should be derived without knowledge of the testing dataset, but then applied to the testing set.
// Common mistake: Allowing the testing dataset to inform transformations made to the training dataset.
// Exceptions: Linear scaling by a constant is permissible (e.g., division by 255).
// Use min/max values or other methods like average and standard deviation for scaling (when we have extreme outliers)

// Note: 
// Testing data may not fit neatly within the scaled bounds of the training data. 
// In case where we are scaling by considering the values in training dataset/ only informing transformations based on training dataset 
// which is the correct method of scaling except when we use some constant to scale linearly in that case we scale train and test dataset by same constant

int main() {

    vector<vector<vector<double>>> X, X_test;
    vector<vector<double>> X_reshaped, X_test_reshaped, X_shuffled;
    vector<unsigned char> temp_2d;
    vector<double> y, y_test, temp, y_shuffled;

    // Data Loading
    create_data_mnist(X, X_test, y, y_test, "F:\\Codes\\Neural Networks From Scratch\\Github Repo\\A Real Dataset\\fashion_mnist_images\\");
    
    cout << X.size() << " " << X[0].size() << " " << X[0][0].size() << "\n";
    cout<<y.size()<<"\n";
    for(int i = 0; i < X.size(); i += 6000) cout<<y[i]<<"\n";
    cout << X_test.size() << X_test[0].size() << X_test[0][0].size()<<"\n";
    cout <<y_test.size()<<"\n";
    
    // Data Preprocessing
    // We are scaling within values -1 to 1
    for(int i = 0; i < X.size(); i++){
        temp.clear();
        for(int j = 0; j < X[0].size(); j++){
            for(int k = 0; k < X[0][0].size(); k++){
                X[i][j][k] = (X[i][j][k] - 127.5) / 127.5;
                // We need to reshape the dataset because our dense layer expects inputs as 2D Matrix and not 3D
                temp.push_back(X[i][j][k]);
            }
        }
        X_reshaped.push_back(temp);
    }

    for(int i = 0; i < X_test.size(); i++){
        temp.clear();
        for(int j = 0; j < X_test[0].size(); j++){
            for(int k = 0; k < X_test[0][0].size(); k++){
                X_test[i][j][k] = (X_test[i][j][k] - 127.5) / 127.5;
                temp.push_back(X_test[i][j][k]);
            }
        }
        X_test_reshaped.push_back(temp);
    }

    cout<<X_reshaped.size() << X_reshaped[0].size()<<"\n";
    cout<<X_test_reshaped.size() << X_test_reshaped[0].size()<<"\n";

    // There are neural network models called convolutional neural networks that will allow you
    // to pass 2D image data “as is,” but a dense neural network like we have here expects samples that
    // are 1D. Even in convolutional neural networks, you will usually need to flatten data before
    // feeding them to an output layer or a dense layer.

    // Data Shuffling

    // Ensure that the dataset is shuffled before training to prevent the model from learning spurious patterns 
    // due to the ordered arrangement of samples and their target classifications.
    // Training on unshuffled data can lead to:
    // 1. The model becoming biased towards predicting the same class within initial batches.
    // 2. Loss spikes and poor performance as the model encounters different classes in subsequent batches.
    // 3. Difficulty in finding a global minimum due to cycling between local minimums for each class.
    // Shuffling the data ensures that each batch contains a mix of different classes, promoting better generalization.
    // When shuffling, ensure that both the samples and their corresponding targets are shuffled in unison.
    // We didn't shuffle the dataset in our previous models because it was small dataset and also we were training by using 
    // all samples in single time but in this case we will be training in batches because the dataset is large.
    // Shuffling a dataset is generally a good practice in machine learning and neural network training.

    temp.clear();
    for(int i = 0; i < y.size(); i++) temp.push_back(i);

    mt19937 rand(0);
    shuffle(temp.begin(), temp.end(), rand);  // shuffling indices to further shuffle X(samples) and y(labels) in unison.

    for(int i = 0; i < temp.size(); i++){
        X_shuffled.push_back(X_reshaped[temp[i]]);
        y_shuffled.push_back(y[temp[i]]);
    }

    // If the model does not train or appears to be misbehaving, you will want to double-check how you preprocessed the data.
    // check if the shuffling is correct or not
    for(int i = 0; i < X_shuffled[0].size(); i++) temp_2d.push_back(static_cast<unsigned char>(round((X_shuffled[4][i] * 127.5) + 127.5)));
    plt::imshow(temp_2d.data(), 28, 28, 1); // The image might be looking slightly different because we scaled the data and then shuffled it.
    plt::show();
    cout<<"label after shuffling: "<< y[4]<<"\n";

    return 0;
}