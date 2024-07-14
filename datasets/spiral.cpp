void create_data(MatrixXd& X, VectorXd& y, long long samples, int classes) {
    
    X.resize(samples * classes, 2);
    y.resize(samples * classes);

    // Random number generator for normal distribution
    mt19937 generator; // Mersenne Twister
    normal_distribution<double> distribution(0.0, 1.0);

    #pragma omp parallel for
    for (size_t i = 0; i < classes; i++) {
        VectorXd random_numbers(samples);

        for (size_t j = 0; j < samples; ++j)
            random_numbers(j) = distribution(generator) * 0.2;

        VectorXd r = VectorXd::LinSpaced(samples, 0.0, 1.0);
        VectorXd t = VectorXd::LinSpaced(samples, i * 4.0, (i + 1) * 4.0) + random_numbers;

        X.block(i * samples, 0, samples, 1) = r.array() * (t.array() * 2.5).sin();
        X.block(i * samples, 1, samples, 1) = r.array() * (t.array() * 2.5).cos();
        
        y.segment(i * samples, samples).setConstant(i);
    }
}
