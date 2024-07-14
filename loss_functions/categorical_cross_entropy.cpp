#include "loss.cpp"

class Loss_CategoricalCrossentropy : public Loss{
    public:
    MatrixXd dinputs;

    VectorXd forward(MatrixXd& y_pred, VectorXd& y_true){

        double lower_limit = 1e-7, upper_limit = 1 - 1e-7;
        MatrixXd clipped = y_pred.cwiseMax(lower_limit).cwiseMin(upper_limit);
        VectorXd correct_confidences(y_true.rows()), neg_log_likelihoods = correct_confidences;

        #pragma omp parallel for
        for(size_t i = 0; i < y_true.rows(); i++)
            correct_confidences(i) = clipped(i, static_cast<int>(y_true(i)));
        
        neg_log_likelihoods = -correct_confidences.array().log();
        return neg_log_likelihoods;
    }

    void backward(MatrixXd& dvalues, VectorXd& y_true){

        this->dinputs = dvalues; // same size
        MatrixXd one_hot_y_true = MatrixXd::Zero(dvalues.rows(), dvalues.cols());

        #pragma omp parallel for
        for (size_t i = 0; i < dvalues.rows(); i++)
            this->dinputs(i, static_cast<int>(y_true(i))) = -1.0 / dvalues(i, static_cast<int>(y_true(i)));
        
        this->dinputs = this->dinputs / dvalues.rows();
    }
};