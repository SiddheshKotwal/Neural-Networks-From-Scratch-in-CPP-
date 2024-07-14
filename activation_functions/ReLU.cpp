class Activation_ReLU{
    public:

    MatrixXd inputs, output, dinputs;

    void forward(MatrixXd& inputs){
        this->inputs = inputs;
        this->output = (inputs.array() > 0).select(inputs.array(), 0);
    }

    void backward(MatrixXd& dvalues){
        this->dinputs = dvalues;
        this->dinputs = (this->inputs.array() > 0).select(this->dinputs.array(), 0);
    }
};