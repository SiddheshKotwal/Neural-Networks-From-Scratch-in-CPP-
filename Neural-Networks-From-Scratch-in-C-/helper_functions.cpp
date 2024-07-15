
// Printing Matrix
void print(vector<vector<double>>& matrix) {
    for (auto i : matrix) {
        for (auto j : i) cout << j << " ";
        cout << "\n";
    }
    cout << "\n";
}

void print(vector<vector<double>>& matrix, string str) {
    cout << str << ":\n";
    for (auto i : matrix) {
        for (auto j : i) cout << j << " ";
        cout << "\n";
    }
    cout << "\n";
}

// Function overloaded for printing vector
void print(vector<double>& vec) {
    for (auto i : vec) cout << i << " ";
    cout << "\n";
}

// Sparse Target classes
double accuracy(vector<vector<double>>& predictions, vector<double>& class_targets) {
    numpy np;
    long long total_correct_predictions = 0;
    vector<double> pred_classes(predictions.size(), 0);
    vector<double> target_classes = class_targets;
    pred_classes = np.argmax(predictions);

    // Calculating total correct predictions in parallel
    #pragma omp parallel for reduction(+:total_correct_predictions)
    for (int i = 0; i < predictions.size(); i++) {
        if (pred_classes[i] == target_classes[i]) total_correct_predictions++;
    }

    return (double)total_correct_predictions / predictions.size();
}

// Function overloading in case of one-hot encoded class target values
double accuracy(vector<vector<double>>& predictions, vector<vector<double>>& class_targets) {
    numpy np;
    long long total_correct_predictions = 0;
    vector<double> pred_classes(predictions.size(), 0), target_classes(predictions.size(), 0);
    pred_classes = np.argmax(predictions);
    target_classes = np.argmax(class_targets);

    // Calculating total correct predictions in parallel
    #pragma omp parallel for reduction(+:total_correct_predictions)
    for (int i = 0; i < predictions.size(); i++) {
        if (pred_classes[i] == target_classes[i]) total_correct_predictions++;
    }

    return (double)total_correct_predictions / predictions.size();
}

double regression_accuracy(vector<vector<double>> y_pred, vector<vector<double>> y_true) {
    numpy np;
    double accuracy_precision = np.stddev(y_true) / 250, total = 0;

    // Calculating accuracy in parallel
    #pragma omp parallel for reduction(+:total)
    for (int i = 0; i < y_pred.size(); i++) {
        if (abs(y_pred[i][0] - y_true[i][0]) < accuracy_precision) total++;
    }
    return total / y_pred.size();
}

vector<double> get_vector(vector<vector<double>>& vec) {
    vector<double> result(vec.size(), 0);
    for (int i = 0; i < vec.size(); i++)
        result[i] = vec[i][0];
    return result;
}
