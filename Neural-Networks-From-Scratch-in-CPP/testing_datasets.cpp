#include "common_includes.h"

// Run Command:
// g++ testing_datasets.cpp -I "C:\Program Files\WindowsApps\PythonSoftwareFoundation.Python.3.9_3.9.3568.0_x64__qbz5n2kfra8p0\Include" -I "c:\users\lenovo\appdata\local\packages\pythonsoftwarefoundation.python.3.9_qbz5n2kfra8p0\localcache\local-packages\python39\site-packages\numpy\core\include" -L "C:\Program Files\WindowsApps\PythonSoftwareFoundation.Python.3.9_3.9.3568.0_x64__qbz5n2kfra8p0\libs" -I "C:\mingw64\mingw64\bin" -lPython39

// use this before running plotting exe file in cmd. 
// Here I have created a bat file and by running bat file this will get executed first than my plotting file and will give output directly
// set PYTHONPATH=C:\Users\Lenovo\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.9_qbz5n2kfra8p0\LocalCache\local-packages\Python39\site-packages

// Approximating the random.randn function of python which generates random numbers from gaussian distribution of mean 0 and variance 1
// Generating dataset containing spiral patterns

int main() {

    // 100 samples for each class meaning generating 300 hundred rows
    long long samples = 100, classes = 3;
    vector<vector<double>> X;
    vector<double> y, x1, x2;
    spiral_data(X, y, samples, classes);

    for (auto i : X) {
        x1.push_back(i[0]);
        x2.push_back(i[1]);
    }
    cout<<x1.size();

    plt::scatter_colored(x1, x2, y, 10.0);
    plt::show();
    return 0;
}
