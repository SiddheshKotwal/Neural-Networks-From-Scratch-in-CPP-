class Activation_Softmax_Loss_CategoricalCrossentropy{
    public:
    MatrixXd dinputs, output;
    Activation_Softmax activation;
    Loss_CategoricalCrossentropy loss_function;

    double forward(MatrixXd& inputs, VectorXd& y_true){
        activation.forward(inputs);
        this->output = activation.output;
        VectorXd pred_losses = loss_function.forward(this->output, y_true);
        return loss_function.calculate(pred_losses);
    }

    void backward(MatrixXd& dvalues, VectorXd& y_true){

        this->dinputs.resize(dvalues.rows(), dvalues.cols());
        #pragma omp parallel for
        for(size_t i = 0; i < dvalues.rows(); i++)
            this->dinputs(i, static_cast<int>(y_true(i))) -= 1;
        this->dinputs /= dvalues.rows();    // Normalize
    }
};