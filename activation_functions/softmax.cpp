class Activation_Softmax{
    public:
    MatrixXd inputs, output, dinputs;

    void forward(MatrixXd& inputs){
        this->inputs = inputs;
        this->output.resize(inputs.rows(), inputs.cols());
        MatrixXd exp_values = inputs;

        #pragma omp parallel for
        for(size_t i = 0; i < inputs.rows(); i++){
            double max_val = inputs.row(i).maxCoeff();
            exp_values(i) = (inputs.row(i).array() - max_val).exp();
        }

        #pragma omp parallel for
        for(size_t i = 0; i < inputs.rows(); i++){
            double sum_exp = exp_values.row(i).sum();
            this->output(i) = exp_values.row(i).array() / sum_exp;
        }
    }

    void backward(MatrixXd& dvalues){
        this->dinputs.resize(dvalues.rows(), dvalues.cols());

        #pragma omp parallel for
        for(size_t i = 0; i < this->output.rows(); i++){
            MatrixXd single_output = this->output.row(i).asColumn();
            MatrixXd jacobian_matrix = this->output.row(i).asDiagonal() - (single_output * single_output.transpose());
            this->dinputs.row(i) = jacobian_matrix * dvalues.row(i).transpose();
        }
    }
};