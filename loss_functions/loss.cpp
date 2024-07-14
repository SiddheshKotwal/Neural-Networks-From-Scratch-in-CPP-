class Loss{
    public:

    double regularization_loss(Layer_Dense& layer){

        double regularization_loss = 0;
        if(layer.weight_regularizer_l1 > 0)
            regularization_loss += layer.weight_regularizer_l1 * layer.weights.array().abs().sum();

        if(layer.weight_regularizer_l2 > 0)
            regularization_loss += layer.weight_regularizer_l2 * layer.weights.array().square().sum();

        if(layer.bias_regularizer_l1 > 0)
            regularization_loss += layer.bias_regularizer_l1 * layer.biases.array().abs().sum();

        if(layer.bias_regularizer_l2 > 0)
            regularization_loss += layer.bias_regularizer_l2 * layer.biases.array().square().sum();

        return regularization_loss;
    }

    double calculate(VectorXd& output){
        return output.mean();
    }
};