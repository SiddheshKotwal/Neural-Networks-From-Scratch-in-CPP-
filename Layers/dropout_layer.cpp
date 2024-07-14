class Layer_Dropout{
    public:
    double rate;
    MatrixXd output, inputs, dinputs, binary_mask;

    Layer_Dropout(double rate){
        this->rate = 1 - rate;
    }

    void forward(MatrixXd& inputs) {

        this->inputs = inputs;
        binary_mask = MatrixXd::Zero(inputs.rows(), inputs.cols());

        // Random number generation
        VectorXd rand_values(inputs.rows() * inputs.cols());
        mt19937 gener(0);
        bernoulli_distribution distri(rate);

        #pragma omp parallel for
        for (int i = 0; i < rand_values.size(); ++i)
            rand_values[i] = distri(gener);

        // Reshape into a matrix
        binary_mask = Map<MatrixXd>(rand_values.data(), inputs.rows(), inputs.cols());

        // Scale the mask and apply it
        binary_mask /= rate;
        this->output = inputs.array() * binary_mask.array();
    }

    void backward(MatrixXd& dvalues){
        this->dinputs = dvalues.array() * this->binary_mask.array();
    }
};