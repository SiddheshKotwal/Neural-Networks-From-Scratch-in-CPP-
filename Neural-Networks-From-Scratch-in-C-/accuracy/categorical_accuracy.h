#ifndef categorical_accuracy
#define categorical_accuracy

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

class Accuracy_Categorical {
public:
    // Sparse Target classes
    double calculate(vector<vector<double>>& predictions, vector<double>& class_targets) {
        numpy np;
        long long total_correct_predictions = 0;
        vector<double> pred_classes(predictions.size(), 0);
        vector<double> target_classes = class_targets;
        pred_classes = np.argmax(predictions);

        // Calculating total correct predictions in parallel
        #pragma omp parallel for reduction(+:total_correct_predictions)
        for (int i = 0; i < predictions.size(); i++) {
            if (pred_classes[i] == target_classes[i]) total_correct_predictions++;
        }

        return (double)total_correct_predictions / predictions.size();
    }

    // Function overloading in case of one-hot encoded class target values
    double calculate(vector<vector<double>>& predictions, vector<vector<double>>& class_targets) {
        numpy np;
        long long total_correct_predictions = 0;
        vector<double> pred_classes(predictions.size(), 0), target_classes(predictions.size(), 0);
        pred_classes = np.argmax(predictions);
        target_classes = np.argmax(class_targets);

        // Calculating total correct predictions in parallel
        #pragma omp parallel for reduction(+:total_correct_predictions)
        for (int i = 0; i < predictions.size(); i++) {
            if (pred_classes[i] == target_classes[i]) total_correct_predictions++;
        }

        return (double)total_correct_predictions / predictions.size();
    }
};

#endif // categorical_accuracy
