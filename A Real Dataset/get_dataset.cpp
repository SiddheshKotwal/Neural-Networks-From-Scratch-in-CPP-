#include "F:\Codes\Neural Networks From Scratch\Github Repo\Neural-Networks-From-Scratch-in-CPP\common_includes.h"
#include <opencv2/opencv.hpp>

int main() {
    cv::Mat image = cv::imread("F:/Codes/Neural Networks From Scratch/Github Repo/A Real Dataset/fashion_mnist_images/train/7/0002.png", cv::IMREAD_UNCHANGED);
    if (image.empty()) {
        std::cerr << "Could not open or find the image!" << std::endl;
        return -1;
    }

    // Check if the image is grayscale or colored
    if (image.channels() == 1) { // Grayscale
        plt::imshow(image.data, image.rows, image.cols, 1); // 1 channel
    } else if (image.channels() == 3) { // RGB
        plt::imshow(image.data, image.rows, image.cols, 3); // 3 channels
    } else {
        std::cerr << "Unsupported image format!" << std::endl;
        return -1;
    }

    plt::show();

    return 0;
}
