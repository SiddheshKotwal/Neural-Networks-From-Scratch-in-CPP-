#include "F:\Codes\Neural Networks From Scratch\Github Repo\Neural-Networks-From-Scratch-in-CPP\common_includes.h"
#include <opencv2/opencv.hpp>
using namespace cv;

int main(){

    Mat image_data = imread("fashion_mnist_images/train/7/0002.png", IMREAD_UNCHANGED);

    if (image_data.empty()) {
        cout << "Could not open or find the image" << endl;
        return -1;
    }

    namedWindow("Display Image", WINDOW_AUTOSIZE);
    imshow("Display Image", image_data);
    waitKey(0);

    return 0;
}
