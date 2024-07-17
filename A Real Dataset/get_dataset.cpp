#include "F:\Codes\Neural Networks From Scratch\Github Repo\Neural-Networks-From-Scratch-in-CPP\common_includes.h"
#include <opencv2/opencv.hpp>

int main() {

    cv::Mat image = cv::imread("F:/Codes/Neural Networks From Scratch/Github Repo/A Real Dataset/fashion_mnist_images/train/7/0002.png", cv::IMREAD_UNCHANGED); // Load an image
    if (image.empty()) {
        std::cerr << "Could not open or find the image!" << std::endl;
        return -1;
    }
    // cout<<image<<" \n";
    cv::imshow("Display window", image); // Show the image in a window
    cv::waitKey(0); // Wait for a keystroke in the window
    return 0;
}
