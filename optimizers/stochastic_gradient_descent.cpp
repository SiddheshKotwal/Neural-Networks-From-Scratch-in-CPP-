class Optimizer_SGD{
    public:
    double learning_rate, decay, momentum, current_learning_rate, iterations;

    Optimizer_SGD(double learning_rate = 1.0, double decay = 0, double momentum = 0)
    :   learning_rate(learning_rate),
        current_learning_rate(learning_rate),
        decay(decay),
        momentum(momentum),
        iterations(0){}

    void pre_update_params(){
        if(this->decay)
            this->current_learning_rate = this->learning_rate * (1.0 / (1.0 + this->decay * this->iterations));
    }

    void update_params(Layer_Dense layer){

        MatrixXd weight_updates(layer.weights.rows(), layer.weights.cols());
        MatrixXd bias_updates(layer.biases.rows(), layer.biases.cols());
        
        if(this->momentum){
            weight_updates = this->momentum * layer.weights_momentums - this->current_learning_rate * layer.dweights;
            layer.weights_momentums = weight_updates;

            bias_updates = this->momentum * layer.biases_momentums - this->current_learning_rate * layer.dbiases;
            layer.biases_momentums = bias_updates;
        }
        else{
            weight_updates = -this->current_learning_rate * layer.dweights;
            bias_updates = -this->current_learning_rate * layer.dbiases;
        }

        layer.weights += weight_updates;
        layer.biases += bias_updates;
    }

    void post_update_params(){
        this->iterations++;
    }
};