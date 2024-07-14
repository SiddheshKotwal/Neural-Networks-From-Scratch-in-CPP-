#include "commonIncludes.h"
#include "matplotlibcpp.h"
#include "datasets/spiral.cpp"
namespace plt = matplotlibcpp;

// ctrl + shift + B to compile and ./run_my_app to execute

int main() {

    long long samples = 100, classes = 3;
    MatrixXd X;
    VectorXd y;
    vector<double> x1, x2, y1;
    create_data(X, y, samples, classes);
    for (size_t i = 0; i < samples * classes ; i++) {
        x1.push_back(X(i, 0));
        x2.push_back(X(i, 1));
        y1.push_back(y(i));
    }

    plt::scatter_colored(x1, x2, y1, 10.0);
    plt::show();
    return 0;
}
