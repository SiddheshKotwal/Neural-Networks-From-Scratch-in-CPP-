#ifndef regression_accuracy
#define regression_accuracy

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

class Accuracy_Regression {
public:
    double calculate(vector<vector<double>>& y_pred, vector<vector<double>>& y_true) {
        numpy np;
        double accuracy_precision = np.stddev(y_true) / 250, total = 0;

        // Calculating accuracy in parallel
        #pragma omp parallel for reduction(+:total)
        for (int i = 0; i < y_pred.size(); i++) {
            if (abs(y_pred[i][0] - y_true[i][0]) < accuracy_precision) total++;
        }

        return total / y_pred.size();
    }
};

#endif // regression_accuracy
