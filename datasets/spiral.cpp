#include <cmath>
#include <cstdlib>
#include <random>
using namespace std;

void spiral_data(vector<vector<double>>& X, vector<double>& y, long long samples, int classes){

    X.resize(samples * classes, vector<double>(2, 0));
    y.resize(samples * classes, 0);

    // Set up the random number generator for normal distribution with mean 0 and variance 1
    mt19937 gen(0);
    normal_distribution<> d(0, 1);

    for(int i = 0; i < classes; i++){

        vector<double> random_numbers(samples, 0);
        double slices_r = 1.0 / samples, r = 0, t = i * 4.0;
        double slices_t = (4.0 / samples);
        long long m = i * samples, n = (i + 1) * samples, k = 0;

        for(int j = 0; j < samples; j++)
            random_numbers[j] = d(gen) * 0.2;

        for(int j = m; j < n; j++){

            X[j][0] = r * sin((t + random_numbers[k]) * 2.5);
            X[j][1] = r * cos((t + random_numbers[k]) * 2.5);
            y[j] = i;
            r += slices_r;
            t += slices_t;
            k++;
        }
    }
}
