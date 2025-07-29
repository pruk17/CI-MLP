#include <iostream> //std::cout, std::cin
#include <vector>
#include <cmath> //sin(), exp(), sqrt()
#include <fstream>
#include <sstream>
#include <random>
#include <algorithm>
#include <iomanip>
#include <string>

using namespace std;

// Configurable parameters
struct Config {
    int input_size;
    int hidden_size;
    int output_size;
    double learning_rate;
    double momentum;
    int epochs;
    int folds;
    int random_seed;
};

// Activation functions
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-max(-500.0, min(500.0, x))));
}

double sigmoid_derivative(double x) {
    return x * (1.0 - x);
}

// Data normalization
void normalize_minmax(vector<vector<double>>& data) {
    if (data.empty()) return;
    
    for (size_t col = 0; col < data[0].size(); ++col) {
        double min_val = data[0][col], max_val = data[0][col];
        for (size_t row = 0; row < data.size(); ++row) {
            min_val = min(min_val, data[row][col]);
            max_val = max(max_val, data[row][col]);
        }
        double range = max_val - min_val;
        if (range == 0) range = 1;
        for (size_t row = 0; row < data.size(); ++row) {
            data[row][col] = (data[row][col] - min_val) / range;
        }
    }
}

void normalize_vector(vector<double>& vec) {
    if (vec.empty()) return;
    
    double min_val = *min_element(vec.begin(), vec.end());
    double max_val = *max_element(vec.begin(), vec.end());
    double range = max_val - min_val;
    if (range == 0) range = 1;
    for (auto& v : vec) {
        v = (v - min_val) / range;
    }
}

// Fixed Multi-Layer Perceptron
class MLP {
private:
    Config config;
    vector<vector<double>> w_ih;  // input to hidden weights
    vector<double> b_h;           // hidden biases
    vector<vector<double>> w_ho;  // hidden to output weights (FIXED: now 2D)
    vector<double> b_o;           // output biases (FIXED: now vector)
    
    // Momentum terms
    vector<vector<double>> delta_w_ih;
    vector<double> delta_b_h;
    vector<vector<double>> delta_w_ho;  // FIXED: now 2D
    vector<double> delta_b_o;           // FIXED: now vector
    
    mt19937 rng;

public:
    MLP(const Config& cfg) : config(cfg), rng(cfg.random_seed) {
        initialize_weights();
    }
    
    void initialize_weights() {
        // Initialize weights and biases
        w_ih.resize(config.hidden_size, vector<double>(config.input_size));
        b_h.resize(config.hidden_size);
        w_ho.resize(config.output_size, vector<double>(config.hidden_size)); // FIXED
        b_o.resize(config.output_size); // FIXED
        
        // Initialize momentum terms
        delta_w_ih.resize(config.hidden_size, vector<double>(config.input_size, 0.0));
        delta_b_h.resize(config.hidden_size, 0.0);
        delta_w_ho.resize(config.output_size, vector<double>(config.hidden_size, 0.0)); // FIXED
        delta_b_o.resize(config.output_size, 0.0); // FIXED
        
        // Xavier initialization
        double limit_ih = sqrt(6.0 / (config.input_size + config.hidden_size));
        double limit_ho = sqrt(6.0 / (config.hidden_size + config.output_size));
        
        uniform_real_distribution<double> dist_ih(-limit_ih, limit_ih);
        uniform_real_distribution<double> dist_ho(-limit_ho, limit_ho);
        
        // Initialize input to hidden weights
        for (int j = 0; j < config.hidden_size; ++j) {
            for (int i = 0; i < config.input_size; ++i) {
                w_ih[j][i] = dist_ih(rng);
            }
            b_h[j] = dist_ih(rng);
        }
        
        // Initialize hidden to output weights
        for (int k = 0; k < config.output_size; ++k) {
            for (int j = 0; j < config.hidden_size; ++j) {
                w_ho[k][j] = dist_ho(rng);
            }
            b_o[k] = dist_ho(rng);
        }
    }
    
    vector<double> forward(const vector<double>& input, vector<double>& hidden_out) {
        // Forward pass through hidden layer
        for (int j = 0; j < config.hidden_size; ++j) {
            double sum = b_h[j];
            for (int i = 0; i < config.input_size; ++i) {
                sum += input[i] * w_ih[j][i];
            }
            hidden_out[j] = sigmoid(sum);
        }
        
        // Forward pass through output layer
        vector<double> output(config.output_size);
        for (int k = 0; k < config.output_size; ++k) {
            double sum = b_o[k]; // FIXED: use correct bias
            for (int j = 0; j < config.hidden_size; ++j) {
                sum += hidden_out[j] * w_ho[k][j]; // FIXED: use correct weight
            }
            output[k] = sigmoid(sum);
        }
        
        return output;
    }
    
    void train(const vector<vector<double>>& X, const vector<vector<double>>& Y) {
        vector<double> hidden(config.hidden_size);
        
        for (int epoch = 0; epoch < config.epochs; ++epoch) {
            double total_error = 0.0;
            
            for (size_t idx = 0; idx < X.size(); ++idx) {
                const auto& x = X[idx];
                const auto& target = Y[idx];
                
                // Forward pass
                vector<double> output = forward(x, hidden);
                
                // Calculate error
                double sample_error = 0.0;
                for (int k = 0; k < config.output_size; ++k) {
                    double error = target[k] - output[k];
                    sample_error += error * error;
                }
                total_error += sample_error;
                
                // Backward pass - Calculate output layer deltas
                vector<double> delta_output(config.output_size);
                for (int k = 0; k < config.output_size; ++k) {
                    double error = target[k] - output[k];
                    delta_output[k] = error * sigmoid_derivative(output[k]);
                }
                
                // Calculate hidden layer deltas
                vector<double> delta_hidden(config.hidden_size);
                for (int j = 0; j < config.hidden_size; ++j) {
                    double error_sum = 0.0;
                    for (int k = 0; k < config.output_size; ++k) {
                        error_sum += delta_output[k] * w_ho[k][j]; // FIXED
                    }
                    delta_hidden[j] = sigmoid_derivative(hidden[j]) * error_sum;
                }
                
                // Update input to hidden weights
                for (int j = 0; j < config.hidden_size; ++j) {
                    for (int i = 0; i < config.input_size; ++i) {
                        double gradient = config.learning_rate * delta_hidden[j] * x[i];
                        double delta = gradient + config.momentum * delta_w_ih[j][i];
                        w_ih[j][i] += delta;
                        delta_w_ih[j][i] = delta;
                    }
                    // Update hidden bias
                    double gradient = config.learning_rate * delta_hidden[j];
                    double delta = gradient + config.momentum * delta_b_h[j];
                    b_h[j] += delta;
                    delta_b_h[j] = delta;
                }
                
                // Update hidden to output weights - FIXED
                for (int k = 0; k < config.output_size; ++k) {
                    for (int j = 0; j < config.hidden_size; ++j) {
                        double gradient = config.learning_rate * delta_output[k] * hidden[j];
                        double delta = gradient + config.momentum * delta_w_ho[k][j];
                        w_ho[k][j] += delta;
                        delta_w_ho[k][j] = delta;
                    }
                    // Update output bias - FIXED
                    double gradient = config.learning_rate * delta_output[k];
                    double delta = gradient + config.momentum * delta_b_o[k];
                    b_o[k] += delta;
                    delta_b_o[k] = delta;
                }
            }
            
            if (epoch % 100 == 0) {
                cout << "Epoch " << epoch << ", MSE: " << total_error / X.size() << endl;
            }
        }
    }
    
    vector<double> predict(const vector<double>& input) {
        vector<double> hidden(config.hidden_size);
        return forward(input, hidden);
    }
};

// Confusion Matrix for classification
void print_confusion_matrix(const vector<vector<double>>& predictions, 
                          const vector<vector<double>>& targets, 
                          int num_classes) {
    vector<vector<int>> matrix(num_classes, vector<int>(num_classes, 0));
    
    for (size_t i = 0; i < predictions.size(); ++i) {
        int pred_class = 0, true_class = 0;
        
        // Find predicted class (highest probability)
        for (int j = 1; j < num_classes; ++j) {
            if (predictions[i][j] > predictions[i][pred_class]) {
                pred_class = j;
            }
        }
        
        // Find true class
        for (int j = 1; j < num_classes; ++j) {
            if (targets[i][j] > targets[i][true_class]) {
                true_class = j;
            }
        }
        
        matrix[true_class][pred_class]++;
    }
    
    cout << "\nConfusion Matrix:" << endl;
    cout << "Actual\\Predicted\t";
    for (int i = 0; i < num_classes; ++i) {
        cout << "Class" << i << "\t";
    }
    cout << endl;
    
    for (int i = 0; i < num_classes; ++i) {
        cout << "Class" << i << "\t\t";
        for (int j = 0; j < num_classes; ++j) {
            cout << matrix[i][j] << "\t";
        }
        cout << endl;
    }
    
    // Calculate accuracy
    int correct = 0, total = 0;
    for (int i = 0; i < num_classes; ++i) {
        correct += matrix[i][i];
        for (int j = 0; j < num_classes; ++j) {
            total += matrix[i][j];
        }
    }
    cout << "Accuracy: " << (double)correct / total * 100 << "%" << endl;
}

// Load flood dataset
bool load_flood_data(const string& filename, vector<vector<double>>& X, vector<vector<double>>& Y) {
    ifstream file(filename);
    if (!file.is_open()) {
        cout << "Error: Cannot open file " << filename << endl;
        return false;
    }
    
    string line;
    while (getline(file, line)) {
        if (line.empty()) continue;
        
        istringstream iss(line);
        vector<double> row(9);
        bool valid = true;
        
        for (int i = 0; i < 9; ++i) {
            if (!(iss >> row[i])) {
                valid = false;
                break;
            }
        }
        
        if (valid) {
            vector<double> input(row.begin(), row.begin() + 8);
            vector<double> output = {row[8]};
            X.push_back(input);
            Y.push_back(output);
        }
    }
    
    cout << "Loaded " << X.size() << " samples from flood dataset" << endl;
    return true;
}

// Load cross.pat dataset
bool load_cross_data(const string& filename, vector<vector<double>>& X, vector<vector<double>>& Y) {
    ifstream file(filename);
    if (!file.is_open()) {
        cout << "Error: Cannot open file " << filename << endl;
        return false;
    }
    
    string line;
    vector<double> current_input(2);
    vector<double> current_output(2);
    bool has_input = false;
    
    while (getline(file, line)) {
        if (line.empty()) continue;
        
        // Skip pattern headers (lines starting with 'p')
        if (line[0] == 'p') {
            has_input = false;
            continue;
        }
        
        istringstream iss(line);
        
        if (!has_input) {
            // This should be input line (2 features)
            if (iss >> current_input[0] >> current_input[1]) {
                has_input = true;
            }
        } else {
            // This should be output line (2 classes)
            if (iss >> current_output[0] >> current_output[1]) {
                X.push_back(current_input);
                Y.push_back(current_output);
                has_input = false;
            }
        }
    }
    
    cout << "Loaded " << X.size() << " samples from cross dataset" << endl;
    return X.size() > 0;
}

// Cross-validation
void cross_validation(const vector<vector<double>>& X, const vector<vector<double>>& Y, 
                     const Config& config, bool is_classification = false) {
    int fold_size = X.size() / config.folds;
    double total_mse = 0.0;
    double total_accuracy = 0.0;
    
    cout << "\n=== " << config.folds << "-Fold Cross Validation ===" << endl;
    cout << "Config: Hidden=" << config.hidden_size << ", LR=" << config.learning_rate 
         << ", Momentum=" << config.momentum << ", Seed=" << config.random_seed << endl;
    
    for (int k = 0; k < config.folds; ++k) {
        vector<vector<double>> X_train, X_test, Y_train, Y_test;
        
        // Split data
        for (size_t i = 0; i < X.size(); ++i) {
            if (i >= k * fold_size && i < (k + 1) * fold_size) {
                X_test.push_back(X[i]);
                Y_test.push_back(Y[i]);
            } else {
                X_train.push_back(X[i]);
                Y_train.push_back(Y[i]);
            }
        }
        
        // Train model
        MLP model(config);
        model.train(X_train, Y_train);
        
        // Test model
        double mse = 0.0;
        int correct = 0;
        vector<vector<double>> predictions;
        
        for (size_t i = 0; i < X_test.size(); ++i) {
            vector<double> pred = model.predict(X_test[i]);
            predictions.push_back(pred);
            
            // Calculate MSE
            for (size_t j = 0; j < pred.size(); ++j) {
                double error = Y_test[i][j] - pred[j];
                mse += error * error;
            }
            
            // Calculate accuracy for classification
            if (is_classification && config.output_size == 2) {
                int pred_class = (pred[0] > pred[1]) ? 0 : 1;
                int true_class = (Y_test[i][0] > Y_test[i][1]) ? 0 : 1;
                if (pred_class == true_class) correct++;
            }
        }
        
        mse /= (X_test.size() * config.output_size);
        double accuracy = (double)correct / X_test.size() * 100;
        
        total_mse += mse;
        total_accuracy += accuracy;
        
        cout << "Fold " << k + 1 << " - MSE: " << fixed << setprecision(6) << mse;
        if (is_classification) {
            cout << ", Accuracy: " << setprecision(2) << accuracy << "%";
        }
        cout << endl;
        
        // Print confusion matrix for the last fold
        if (is_classification && k == config.folds - 1) {
            print_confusion_matrix(predictions, Y_test, config.output_size);
        }
    }
    
    cout << "\nAverage MSE: " << fixed << setprecision(6) << total_mse / config.folds << endl;
    if (is_classification) {
        cout << "Average Accuracy: " << setprecision(2) << total_accuracy / config.folds << "%" << endl;
    }
}

// Comprehensive parameter testing
void comprehensive_parameter_testing(const vector<vector<double>>& X, const vector<vector<double>>& Y, 
                                   bool is_classification, const string& dataset_name) {
    vector<int> hidden_sizes = {3, 5, 7, 10, 15};
    vector<double> learning_rates = {0.001, 0.01, 0.05, 0.1};
    vector<double> momentum_rates = {0.0, 0.5, 0.9};
    vector<int> seeds = {42, 123, 456};
    
    cout << "\n=== Comprehensive Parameter Testing for " << dataset_name << " ===" << endl;
    cout << "Hidden\tLR\tMomentum\tSeed\tAvg_MSE\t\tAvg_Accuracy" << endl;
    cout << "------\t--\t--------\t----\t-------\t\t------------" << endl;
    
    for (int hidden : hidden_sizes) {
        for (double lr : learning_rates) {
            for (double momentum : momentum_rates) {
                for (int seed : seeds) {
                    Config config = {
                        (int)X[0].size(), hidden, (int)Y[0].size(), 
                        lr, momentum, is_classification ? 500 : 1000, 
                        10, seed
                    };
                    
                    // Quick evaluation without printing details
                    int fold_size = X.size() / config.folds;
                    double total_mse = 0.0;
                    double total_accuracy = 0.0;
                    
                    for (int k = 0; k < config.folds; ++k) {
                        vector<vector<double>> X_train, X_test, Y_train, Y_test;
                        
                        for (size_t i = 0; i < X.size(); ++i) {
                            if (i >= k * fold_size && i < (k + 1) * fold_size) {
                                X_test.push_back(X[i]);
                                Y_test.push_back(Y[i]);
                            } else {
                                X_train.push_back(X[i]);
                                Y_train.push_back(Y[i]);
                            }
                        }
                        
                        MLP model(config);
                        model.train(X_train, Y_train);
                        
                        double mse = 0.0;
                        int correct = 0;
                        
                        for (size_t i = 0; i < X_test.size(); ++i) {
                            vector<double> pred = model.predict(X_test[i]);
                            
                            for (size_t j = 0; j < pred.size(); ++j) {
                                double error = Y_test[i][j] - pred[j];
                                mse += error * error;
                            }
                            
                            if (is_classification && config.output_size == 2) {
                                int pred_class = (pred[0] > pred[1]) ? 0 : 1;
                                int true_class = (Y_test[i][0] > Y_test[i][1]) ? 0 : 1;
                                if (pred_class == true_class) correct++;
                            }
                        }
                        
                        mse /= (X_test.size() * config.output_size);
                        double accuracy = (double)correct / X_test.size() * 100;
                        
                        total_mse += mse;
                        total_accuracy += accuracy;
                    }
                    
                    cout << hidden << "\t" << lr << "\t" << momentum 
                         << "\t\t" << seed << "\t" << fixed << setprecision(6) 
                         << total_mse / config.folds;
                    if (is_classification) {
                        cout << "\t\t" << setprecision(2) << total_accuracy / config.folds << "%";
                    }
                    cout << endl;
                }
            }
        }
    }
}

int main() {
    cout << "=== Enhanced MLP with Fixed Backpropagation ===" << endl;
    
    // Test with flood dataset
    cout << "\n1. Testing with Flood Dataset:" << endl;
    vector<vector<double>> X_flood, Y_flood;
    
    if (load_flood_data("Flood_dataset.txt", X_flood, Y_flood)) {
        normalize_minmax(X_flood);
        
        // Normalize Y manually for regression
        vector<double> Y_flat;
        for (const auto& y : Y_flood) Y_flat.push_back(y[0]);
        normalize_vector(Y_flat);
        for (size_t i = 0; i < Y_flood.size(); ++i) {
            Y_flood[i][0] = Y_flat[i];
        }
        
        // Test different configurations
        vector<Config> flood_configs = {
            {8, 5, 1, 0.01, 0.9, 1000, 10, 42},
            {8, 10, 1, 0.01, 0.9, 1000, 10, 42},
            {8, 15, 1, 0.01, 0.9, 1000, 10, 42},
            {8, 10, 1, 0.001, 0.9, 1000, 10, 42},
            {8, 10, 1, 0.05, 0.9, 1000, 10, 42},
            {8, 10, 1, 0.01, 0.0, 1000, 10, 42},
            {8, 10, 1, 0.01, 0.5, 1000, 10, 42}
        };
        
        for (const auto& config : flood_configs) {
            cross_validation(X_flood, Y_flood, config, false);
        }
        
        // Comprehensive parameter testing
        comprehensive_parameter_testing(X_flood, Y_flood, false, "Flood Dataset");
    }
    
    // Test with cross.pat dataset
    cout << "\n2. Testing with Cross Dataset:" << endl;
    vector<vector<double>> X_cross, Y_cross;
    
    if (load_cross_data("cross.pat", X_cross, Y_cross) && X_cross.size() > 0) {
        normalize_minmax(X_cross);
        
        // Test different configurations with lower learning rates
        vector<Config> cross_configs = {
            {2, 3, 2, 0.01, 0.9, 1000, 10, 42},
            {2, 5, 2, 0.01, 0.9, 1000, 10, 42},
            {2, 7, 2, 0.01, 0.9, 1000, 10, 42},
            {2, 5, 2, 0.001, 0.9, 1000, 10, 42},
            {2, 5, 2, 0.05, 0.9, 1000, 10, 42},
            {2, 5, 2, 0.01, 0.0, 1000, 10, 42},
            {2, 5, 2, 0.01, 0.5, 1000, 10, 42}
        };
        
        for (const auto& config : cross_configs) {
            cross_validation(X_cross, Y_cross, config, true);
        }
        
        // Comprehensive parameter testing
        comprehensive_parameter_testing(X_cross, Y_cross, true, "Cross Dataset");
    } else {
        cout << "Failed to load cross dataset or no data found." << endl;
    }
    
    return 0;
}