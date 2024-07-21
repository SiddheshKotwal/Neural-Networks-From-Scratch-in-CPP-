// common_includes.h
#ifndef COMMON_INCLUDES_H
#define COMMON_INCLUDES_H

// Necessary library headers
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cstdlib>
#include <vector>
#include <tuple>
#include <map>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>
#include <omp.h>
#include "math_operations/numpy_operations.h"
#include "matplotlibcpp.h"
#include "datasets/spiral.h"
#include "datasets/vertical.h"
#include "datasets/sine.h"
#include "Layers/dropout_layer.h"
#include "Layers/dense_layer.h"
#include "activation_functions/ReLU.h"
#include "activation_functions/Softmax.h"
#include "activation_functions/Sigmoid.h"
#include "activation_functions/Linear.h"
#include "loss_functions/loss_functions.h"
#include "loss_functions/categorical_cross_entropy.h"
#include "loss_functions/binary_cross_entropy.h"
#include "loss_functions/mean_squared_error.h"
#include "loss_functions/mean_absolute_error.h"
#include "classifier/softmax_classifier.h"
#include "accuracy/calculate_accuracy.h"
#include "accuracy/binary_logistic_regression_accuracy.h"
#include "accuracy/categorical_accuracy.h"
#include "accuracy/regression_accuracy.h"
#include "optimizer/stochastic_gradient_descent.h"
#include "optimizer/adaptive_gradient.h"
#include "optimizer/root_mean_square_propagation.h"
#include "optimizer/adaptive_moment_estimation.h"
using namespace std;
namespace plt = matplotlibcpp;

#endif // COMMON_INCLUDES_H
