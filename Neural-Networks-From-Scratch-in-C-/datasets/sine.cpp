#include<cmath>
using namespace std;

void sine_data(vector<vector<double>>& X, vector<vector<double>>& y, long long samples = 1000){

    X.resize(samples, vector<double>(1, 0));
    y.resize(samples, vector<double>(1, 0));

    for(int i = 0; i < samples; i++){
        X[i][0] = (double)i / samples;
        y[i][0] = sin(2 * M_PI * X[i][0]);
    }
}