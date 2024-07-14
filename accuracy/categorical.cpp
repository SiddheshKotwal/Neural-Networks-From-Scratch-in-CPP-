class Accuracy_Categorical {
public:
    void init(VectorXd& y) {}

    double calculate(MatrixXd& predictions, VectorXd& y) {
        long long count = 0;

        #pragma omp parallel for reduction(+:count)
        for (size_t i = 0; i < predictions.rows(); i++) {
            int pred_class = predictions.row(i).maxCoeffIndex();
            if (pred_class == y(i)) count++;
        }

        return static_cast<double>(count) / y.rows();
    }
};