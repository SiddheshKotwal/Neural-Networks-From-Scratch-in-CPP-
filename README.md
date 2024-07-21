# Neural Networks from Scratch in C++ and Real Dataset Testing

## Project Overview

This project includes two main directories:

1. **Neural Networks from Scratch in CPP**
2. **A Real Dataset**

These directories cover the implementation and testing of various neural networks.

---

## Directory: Neural Networks from Scratch in CPP

### Overview

This directory contains implementations for:

- **Classification Neural Network**
  - **Activation Functions**: ReLU for hidden layers, Softmax for the output layer.
  - **Loss Function**: Categorical Cross-Entropy.

- **Binary Logistic Regression Neural Network**
  - **Activation Functions**: ReLU for hidden layers, Sigmoid for the output layer.
  - **Loss Function**: Binary Cross-Entropy.

- **Regression Neural Network**
  - **Activation Functions**: ReLU for hidden layers, Linear activation for the output layer.
  - **Loss Functions**: Mean Squared Error (MSE) or Mean Absolute Error (MAE).

### Optimizers

- Stochastic Gradient Descent (SGD) with Momentum
- RMSProp
- AdaGrad
- Adam

### Setup and Compilation

1. **Place the `.vscode` Directory:**
   - Ensure the `.vscode` directory is in the parent directory of your project directory.

2. **Configuration:**
   - Make sure include, bin, and linking paths for necessary libraries (e.g., NumPy, Python, OpenCV, Cereal) are correctly set in the `.vscode` and CMake files.

3. **Compile and Run:**
   - Use `Ctrl + Shift + B` in VSCode to build the project.
   - For applications using `matplotlibcpp`, update the `run_my_app.bat` file with the `.cpp` filename and execute `./run_my_app`.
   - For applications not using `matplotlibcpp`, simply run `./filename`.

---

## Directory: A Real Dataset

### Overview

This directory tests the classification neural network on the Fashion MNIST dataset. 

- **Dataset**: Fashion MNIST
- **Description**: Fashion MNIST consists of 60,000 training samples and 10,000 test samples. Each sample is a 28x28 grayscale image of a fashion item, such as a shirt, dress, or sneaker. The dataset is intended to serve as a more challenging replacement for the classic MNIST dataset of handwritten digits.

### Setup and Compilation

1. **Install Required Tools and Libraries:**

   - **MSYS2 and MinGW-w64:**
     1. Install MSYS2.
     2. Use MSYS2 to install MinGW-w64 with POSIX threads.
     3. Add the MSYS2 MinGW paths to your environment variables.

   - **OpenCV:**
     1. Install the MinGW build of OpenCV.
     2. Add the OpenCV `bin` path to your environment variables.

   - **Cereal:**
     1. Unzip the Cereal library.
     2. Add the Cereal include path to the CMake file.

2. **Compile Using CMake:**

   1. Create and navigate to the `build` directory:
      ```bash
      mkdir build
      cd build
      ```

   2. Configure the project with CMake:
      ```bash
      cmake -DSOURCE_FILE_ARG="your_filename.cpp" -G "MinGW Makefiles" ..
      ```

   3. Build the project:
      ```bash
      cmake --build .
      ```

   4. Navigate to the executable directory and run:
      ```bash
      cd "A real dataset"
      ./run_my_plot (if using matplotlibcpp)
      ./MyProgram.exe (otherwise)
      ```

---

## Configuration Files

The following files are used for configuration:

- **`.vscode/c_cpp_properties.json`**: Configuration for IntelliSense, including include paths and compiler settings.
- **`.vscode/settings.json`**: VSCode settings for file associations and CMake configuration.
- **`.vscode/tasks.json`**: Task configuration for building the project using `g++`.

- **`CMakeLists.txt`**: CMake configuration file for setting up the project, finding libraries, and specifying compiler options.

For detailed paths and settings, refer to the respective configuration files included in the repository.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

Feel free to reach out if you have any questions or need further assistance!
