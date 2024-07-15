#include <cmath>

class numpy {
public:
    // Implementing matrix product
    vector<vector<double>> dot(vector<vector<double>>& matrix1, vector<vector<double>>& matrix2) {
        if (matrix1[0].size() != matrix2.size()) {
            cout << "Matrix product not possible !!\n";
            return {{}};
        }

        vector<vector<double>> result(matrix1.size(), vector<double>(matrix2[0].size()));
        #pragma omp parallel for
        for (int m = 0; m < matrix1.size(); m++) {
            for (int n = 0; n < matrix2[0].size(); n++) {
                double product = 0;
                for (int i = 0; i < matrix2.size(); i++) {
                    product += matrix1[m][i] * matrix2[i][n];
                }
                result[m][n] = product;
            }
        }

        return result;
    }

    // Implementing Transpose of a Matrix
    vector<vector<double>> T(vector<vector<double>>& matrix) {
        vector<vector<double>> result(matrix[0].size(), vector<double>(matrix.size()));
        #pragma omp parallel for
        for (int i = 0; i < matrix[0].size(); i++) {
            for (int j = 0; j < matrix.size(); j++) {
                result[i][j] = matrix[j][i];
            }
        }
        return result;
    }

    // Adding Biases
    vector<vector<double>> addBias(vector<vector<double>>& layer_outputs, vector<double>& biases) {
        #pragma omp parallel for
        for (int i = 0; i < layer_outputs.size(); i++) {
            for (int j = 0; j < layer_outputs[0].size(); j++) {
                layer_outputs[i][j] += biases[j];
            }
        }
        return layer_outputs;
    }

    // Clipping the vector's values below start and above end
    vector<vector<double>> clip(vector<vector<double>>& output, double start, double end) {
        vector<vector<double>> clipped_output(output.size(), vector<double>(output[0].size(), 0));
        #pragma omp parallel for
        for (int i = 0; i < output.size(); i++) {
            for (int j = 0; j < output[0].size(); j++) {
                if (output[i][j] <= 0) clipped_output[i][j] = start;
                else if (output[i][j] >= 1) clipped_output[i][j] = end;
                else clipped_output[i][j] = output[i][j];
            }
        }
        return clipped_output;
    }

    // Calculating Mean of values
    double mean(vector<double>& output) {
        double total = 0;
        #pragma omp parallel for reduction(+:total)
        for (size_t i = 0; i < output.size(); i++) {
            total += output[i];
        }
        return total / output.size();
    }

    // Calculating Mean of matrix
    double mean(vector<vector<double>>& output) {
        double total = 0;
        #pragma omp parallel for reduction(+:total)
        for (size_t i = 0; i < output.size(); i++) {
            for (size_t j = 0; j < output[0].size(); j++) {
                total += output[i][j];
            }
        }
        return total / (output.size() * output[0].size());
    }

    // Calculating negative logarithms of values
    vector<double> negative_log(vector<double>& confidences) {
        vector<double> negative_logs(confidences.size(), 0);
        #pragma omp parallel for
        for (int i = 0; i < confidences.size(); i++) {
            negative_logs[i] = -1 * log(confidences[i]);
        }
        return negative_logs;
    }

    // Finding maximum values and returning their indices
    vector<double> argmax(vector<vector<double>>& matrix) {
        vector<double> max_indices(matrix.size(), 0);
        #pragma omp parallel for
        for (int i = 0; i < matrix.size(); i++) {
            double maxi = matrix[i][0];
            double index = 0;
            for (int j = 1; j < matrix[0].size(); j++) {
                if (maxi < matrix[i][j]) {
                    maxi = matrix[i][j];
                    index = j;
                }
            }
            max_indices[i] = index;
        }
        return max_indices;
    }

    // Summation by row or column axis
    vector<vector<double>> sum(vector<vector<double>>& dvalues, int axis) {
        vector<vector<double>> sum(1, vector<double>(dvalues[0].size(), 0));
        if (axis == 0) {
            // column wise sum -> axis = 0
            #pragma omp parallel for
            for (int i = 0; i < dvalues[0].size(); i++) {
                double total = 0;
                for (int j = 0; j < dvalues.size(); j++) {
                    total += dvalues[j][i];
                }
                sum[0][i] = total;
            }
        } else {
            sum.resize(dvalues.size(), vector<double>(1, 0));
            // row wise sum -> axis = 1
            #pragma omp parallel for
            for (int i = 0; i < dvalues.size(); i++) {
                double total = 0;
                for (int j = 0; j < dvalues[0].size(); j++) {
                    total += dvalues[i][j];
                }
                sum[i][0] = total;
            }
        }
        return sum;
    }

    // Converting column matrix into diagonal matrix
    vector<vector<double>> diagflat(vector<vector<double>>& column_matrix) {
        vector<vector<double>> matrix(column_matrix.size(), vector<double>(column_matrix.size(), 0));
        for (int i = 0; i < matrix.size(); i++)
            matrix[i][i] = column_matrix[i][0];
        return matrix;
    }

    // Subtracting two matrices
    vector<vector<double>> sub(vector<vector<double>>& a, vector<vector<double>>& b) {
        vector<vector<double>> result(a.size(), vector<double>(a[0].size(), 0));
        #pragma omp parallel for
        for (size_t i = 0; i < a.size(); ++i) {
            for (size_t j = 0; j < a[0].size(); ++j) {
                result[i][j] = a[i][j] - b[i][j];
            }
        }
        return result;
    }

    // Calculating standard deviation
    double stddev(vector<vector<double>>& data) {
        double sum = 0.0;
        #pragma omp parallel for reduction(+:sum)
        for (size_t i = 0; i < data.size(); i++) {
            sum += data[i][0]; // Assuming calculating based on first column
        }

        double mean_val = sum / data.size();
        double variance = 0.0;

        #pragma omp parallel for reduction(+:variance)
        for (size_t i = 0; i < data.size(); i++) {
            variance += pow(data[i][0] - mean_val, 2);
        }

        variance /= data.size();
        return sqrt(variance);
    }
};
