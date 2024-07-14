class Layer_Dense{
    
    public:
    MatrixXd weights, biases, output, inputs, dweights, dbiases, dinputs, weights_momentums, biases_momentums, weight_cache, bias_cache;
    double weight_regularizer_l1, weight_regularizer_l2, bias_regularizer_l1, bias_regularizer_l2;

    // Layer Initialization
    Layer_Dense(long long n_inputs, long long n_neurons, double weight_regularizer_l1 = 0, double weight_regularizer_l2 = 0, double bias_regularizer_l1 = 0, double bias_regularizer_l2 = 0)
    :   weights(MatrixXd::Zero(n_inputs, n_neurons)),
        weights_momentums(weights), weight_cache(weights),
        biases(MatrixXd::Zero(1, n_neurons)),
        biases_momentums(biases), bias_cache(biases),
        weight_regularizer_l1(weight_regularizer_l1),
        weight_regularizer_l2(weight_regularizer_l2),
        bias_regularizer_l1(bias_regularizer_l1),
        bias_regularizer_l2(bias_regularizer_l2){

        mt19937 rng(0); // seed 0
        normal_distribution<double> dist(0, 1); // mean = 0, var = 1
        for(size_t i = 0; i < n_inputs; i++){
            for(size_t j = 0; j < n_neurons; j++)
                this->weights(i, j) = dist(rng) * 0.01;
        }
    }

    void forward(MatrixXd& inputs){
        
        this->inputs = inputs;  // for backpropagation
        this->output = inputs * this->weights;
        this->output = this->output.rowwise() + this->biases;
    }

    void backward(MatrixXd& dvalues){

        // Gradient on parameters
        this->dweights = this->inputs.transpose() * dvalues;
        this->dbiases = dvalues.colwise().sum();

        // Gradient on regularization
        if(this->weight_regularizer_l1 > 0){
            MatrixXd dL1 = (this->weights.array() < 0).select(-1, 1);
            this->dweights += this->weight_regularizer_l1 * dL1;
        }

        if(this->weight_regularizer_l2 > 0){
            this->dweights += 2 * this->weight_regularizer_l2 * this->weights;
        }

        if(this->bias_regularizer_l1 > 0){
            MatrixXd dL1 = (this->biases.array() < 0).select(-1, 1);
            this->dbiases += this->bias_regularizer_l1 * dL1;
        }

        if(this->bias_regularizer_l2 > 0){
            this->dbiases += 2 * this->bias_regularizer_l2 * this->biases;
        }

        // Gradient on values
        this->dinputs = dvalues * this->weights.transpose();
    }
};
