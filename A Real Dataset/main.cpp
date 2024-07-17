#include "F:\Codes\Neural Networks From Scratch\Github Repo\Neural-Networks-From-Scratch-in-CPP\common_includes.h"
#include "get_dataset.cpp"

int main() {

    vector<vector<vector<double>>> X, X_test;
    vector<double> y, y_test;

    create_data_mnist(X, X_test, y, y_test, "F:\\Codes\\Neural Networks From Scratch\\Github Repo\\A Real Dataset\\fashion_mnist_images\\");
    
    cout << X.size() << " " << X[0].size() << " " << X[0][0].size() << "\n";
    cout<<y.size()<<"\n";
    for(int i = 0; i < X.size(); i += 6000) cout<<y[i]<<"\n";
    cout << X_test.size() << X_test[0].size() << X_test[0][0].size()<<"\n";
    cout <<y_test.size()<<"\n";
    
    return 0;
}