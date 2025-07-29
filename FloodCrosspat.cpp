#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <cmath>
#include <random>
#include <algorithm>
#include <string>
#include <iomanip> // for std::setw

using namespace std;

// === CONFIG ===
const int INPUT_SIZE = 2;
const int OUTPUT_SIZE = 2;
const int EPOCHS = 500; // training epochs per fold
const int K_FOLD = 10;  // 10-fold cross validation

// Data structure for one sample
struct Sample {
    vector<double> input;   // size = 2 features
    vector<double> output;  // size = 2 classes (one-hot)
};

vector<Sample> dataset;

// === Activation Functions ===
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}
double sigmoid_derivative(double x) {
    return x * (1.0 - x);
}

// === Trim spaces from string ===
string trim(const string& s) {
    size_t start = 0, end = s.size();
    while (start < end && isspace((unsigned char)s[start])) ++start;
    while (end > start && isspace((unsigned char)s[end-1])) --end;
    return s.substr(start, end - start);
}

// === Load cross.csv dataset ===
// Format:
//   2 feature values separated by comma
//   2 class values separated by space
// Repeats for all samples
bool load_dataset(const string& filename) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Cannot open file " << filename << endl;
        return false;
    }

    string line;
    dataset.clear();
    while (getline(file, line)) {
        line = trim(line);
        if (line.empty()) continue;
        if (line[0] == 'p') {
            // label line, ignore (e.g. p0, p1,...)
            continue;
        }

        // read features line
        istringstream ss_feat(line);
        string feat_str;
        vector<double> features;
        while (getline(ss_feat, feat_str, ',')) {
            try {
                features.push_back(stod(trim(feat_str)));
            } catch (...) {
                cerr << "Invalid feature number format: " << line << endl;
                return false;
            }
        }
        if (features.size() != INPUT_SIZE) {
            cerr << "Incorrect feature count: " << features.size() << endl;
            return false;
        }

        // read next line for output classes (one-hot)
        if (!getline(file, line)) {
            cerr << "Missing output class line" << endl;
            return false;
        }
        line = trim(line);
        istringstream ss_out(line);
        vector<double> outputs;
        string out_str;
        while (ss_out >> out_str) {
            try {
                outputs.push_back(stod(out_str));
            } catch (...) {
                cerr << "Invalid output number format: " << line << endl;
                return false;
            }
        }
        if (outputs.size() != OUTPUT_SIZE) {
            cerr << "Incorrect output class count: " << outputs.size() << endl;
            return false;
        }

        dataset.push_back({features, outputs});
    }

    return true;
}

// === Normalize dataset feature values to [0,1] ===
// Outputs assumed one-hot, no need to normalize
void normalize_dataset(vector<Sample>& data) {
    for (int i = 0; i < INPUT_SIZE; ++i) {
        double min_val = 1e9, max_val = -1e9;
        for (const auto& s : data) {
            min_val = min(min_val, s.input[i]);
            max_val = max(max_val, s.input[i]);
        }
        double range = max_val - min_val;
        if (range < 1e-9) range = 1; // avoid divide by zero
        for (auto& s : data) {
            s.input[i] = (s.input[i] - min_val) / range;
        }
    }
}

// === MLP Class ===
class MLP {
public:
    vector<vector<vector<double>>> weights;      // weights[layer][from][to]
    vector<vector<vector<double>>> delta_weights; // momentum term for weight update
    vector<vector<double>> layers;  // activations per layer
    vector<vector<double>> errors;  // errors per layer (excluding input)

    MLP(int input_size, const vector<int>& hidden_sizes, int output_size, mt19937& gen) {
        init_network(input_size, hidden_sizes, output_size, gen);
    }

    // Initialize network weights randomly between -1 and 1
    void init_network(int input_size, const vector<int>& hidden_sizes, int output_size, mt19937& gen) {
        uniform_real_distribution<> dist(-1, 1);
        int prev_size = input_size;
        int total_layers = hidden_sizes.size() + 1; // hidden layers + output layer

        weights.resize(total_layers);
        delta_weights.resize(total_layers);

        for (int l = 0; l < total_layers; ++l) {
            int curr_size = (l == total_layers - 1) ? output_size : hidden_sizes[l];
            weights[l].resize(prev_size, vector<double>(curr_size));
            delta_weights[l].resize(prev_size, vector<double>(curr_size, 0.0));
            for (int i = 0; i < prev_size; ++i)
                for (int j = 0; j < curr_size; ++j)
                    weights[l][i][j] = dist(gen);
            prev_size = curr_size;
        }

        layers.resize(total_layers + 1);
        layers[0].resize(input_size);
        for (int i = 0; i < hidden_sizes.size(); ++i)
            layers[i + 1].resize(hidden_sizes[i]);
        layers[total_layers].resize(output_size);

        errors.resize(total_layers);
        for (int i = 0; i < total_layers; ++i) {
            int sz = (i == total_layers - 1) ? output_size : hidden_sizes[i];
            errors[i].resize(sz);
        }
    }

    // Forward pass: input -> output
    void forward(const vector<double>& input) {
        layers[0] = input;
        for (int l = 0; l < (int)weights.size(); ++l) {
            for (int j = 0; j < (int)layers[l + 1].size(); ++j) {
                double sum = 0.0;
                for (int i = 0; i < (int)layers[l].size(); ++i)
                    sum += layers[l][i] * weights[l][i][j];
                layers[l + 1][j] = sigmoid(sum);
            }
        }
    }

    // Backpropagation: adjust weights according to target output
    void backward(const vector<double>& target, double learning_rate, double momentum) {
        int last = (int)errors.size() - 1;

        // output layer error
        for (int k = 0; k < (int)layers.back().size(); ++k) {
            errors[last][k] = (target[k] - layers.back()[k]) * sigmoid_derivative(layers.back()[k]);
        }

        // hidden layers error backwards
        for (int l = last - 1; l >= 0; --l) {
            for (int i = 0; i < (int)errors[l].size(); ++i) {
                double err_sum = 0.0;
                for (int j = 0; j < (int)errors[l + 1].size(); ++j)
                    err_sum += errors[l + 1][j] * weights[l + 1][i][j];
                errors[l][i] = err_sum * sigmoid_derivative(layers[l + 1][i]);
            }
        }

        // update weights with momentum and learning rate
        for (int l = 0; l < (int)weights.size(); ++l) {
            for (int i = 0; i < (int)weights[l].size(); ++i) {
                for (int j = 0; j < (int)weights[l][i].size(); ++j) {
                    double delta = learning_rate * errors[l][j] * layers[l][i] + momentum * delta_weights[l][i][j];
                    weights[l][i][j] += delta;
                    delta_weights[l][i][j] = delta;
                }
            }
        }
    }

    // Predict class index (argmax output node)
    int predict_class() {
        int idx = 0;
        double max_val = layers.back()[0];
        for (int i = 1; i < (int)layers.back().size(); ++i) {
            if (layers.back()[i] > max_val) {
                max_val = layers.back()[i];
                idx = i;
            }
        }
        return idx;
    }
};

// === Evaluate accuracy and confusion matrix ===
struct ConfusionMatrix {
    int TP, FP, TN, FN;

    ConfusionMatrix() : TP(0), FP(0), TN(0), FN(0) {}

    void update(int true_class, int predicted_class) {
        if (true_class == 1 && predicted_class == 1) TP++;
        else if (true_class == 1 && predicted_class == 0) FN++;
        else if (true_class == 0 && predicted_class == 0) TN++;
        else if (true_class == 0 && predicted_class == 1) FP++;
    }

    double accuracy() const {
        int total = TP + FP + TN + FN;
        if (total == 0) return 0;
        return (double)(TP + TN) / total;
    }

    void print() const {
        cout << "Confusion Matrix:\n";
        cout << "          Predicted\n";
        cout << "          0     1\n";
        cout << "Actual 0  " << setw(5) << TN << "  " << setw(5) << FP << "\n";
        cout << "       1  " << setw(5) << FN << "  " << setw(5) << TP << "\n";
        cout << fixed << setprecision(4);
        cout << "Accuracy: " << accuracy() * 100 << "%\n";
    }
};

// === Main training and evaluation with 10-fold cross validation ===
void k_fold_cross_validation(
    const vector<vector<int>>& hidden_layer_configs,
    const vector<double>& learning_rates,
    const vector<double>& momentum_values,
    int num_trials = 3)
{
    cout << "Total dataset size: " << dataset.size() << "\n";

    random_device rd;
    mt19937 gen(rd());

    struct Result {
        vector<int> hidden_layers;
        double learning_rate;
        double momentum;
        double avg_accuracy;
        double avg_epochs;
    };

    vector<Result> all_results;

    // Iterate over all combinations of hyperparameters
    for (auto& hidden_layers : hidden_layer_configs) {
        for (double lr : learning_rates) {
            for (double mo : momentum_values) {
                double sum_accuracy = 0.0;
                double sum_epochs = 0.0;

                for (int trial = 0; trial < num_trials; ++trial) {
                    // Shuffle dataset for randomness
                    vector<Sample> data_shuffled = dataset;
                    shuffle(data_shuffled.begin(), data_shuffled.end(), gen);

                    // Split into K folds
                    int fold_size = (int)data_shuffled.size() / K_FOLD;

                    double total_accuracy = 0.0;
                    double total_epoch_count = 0;

                    for (int fold = 0; fold < K_FOLD; ++fold) {
                        int start_idx = fold * fold_size;
                        int end_idx = (fold == K_FOLD - 1) ? (int)data_shuffled.size() : start_idx + fold_size;

                        vector<Sample> train_set;
                        vector<Sample> test_set;

                        for (int i = 0; i < (int)data_shuffled.size(); ++i) {
                            if (i >= start_idx && i < end_idx)
                                test_set.push_back(data_shuffled[i]);
                            else
                                train_set.push_back(data_shuffled[i]);
                        }

                        // Create new network instance with random weights for each fold
                        MLP net(INPUT_SIZE, hidden_layers, OUTPUT_SIZE, gen);

                        // Train network for EPOCHS
                        for (int epoch = 0; epoch < EPOCHS; ++epoch) {
                            for (const auto& sample : train_set) {
                                net.forward(sample.input);
                                net.backward(sample.output, lr, mo);
                            }
                        }

                        // Test on test_set, build confusion matrix
                        ConfusionMatrix cm;
                        for (const auto& sample : test_set) {
                            net.forward(sample.input);
                            int pred = net.predict_class();

                            int true_class = (sample.output[0] == 1) ? 0 : 1; // class0 if output[0]==1 else class1
                            cm.update(true_class, pred);
                        }

                        double acc = cm.accuracy();
                        total_accuracy += acc;
                        total_epoch_count += EPOCHS;

                        if (fold == 0 && trial == 0) {
                            cout << "Confusion matrix example for config hidden layers = [";
                            for (auto h : hidden_layers) cout << h << " ";
                            cout << "], lr=" << lr << ", momentum=" << mo << ":\n";
                            cm.print();
                            cout << "\n";
                        }
                    } // end fold

                    sum_accuracy += total_accuracy / K_FOLD;
                    sum_epochs += total_epoch_count / K_FOLD;
                } // end trial

                Result r = {hidden_layers, lr, mo, sum_accuracy / num_trials, sum_epochs / num_trials};
                all_results.push_back(r);
                cout << "Config: Hidden layers = [";
                for (auto h : hidden_layers) cout << h << " ";
                cout << "], LR = " << lr << ", Momentum = " << mo
                     << ", Avg Accuracy = " << r.avg_accuracy * 100 << "%, Avg Epochs = " << r.avg_epochs << "\n";
            }
        }
    }

    // Optionally summarize best results
    cout << "\nSummary of all tested configurations:\n";
    for (const auto& r : all_results) {
        cout << "Hidden Layers: [";
        for (auto h : r.hidden_layers) cout << h << " ";
        cout << "], LR=" << r.learning_rate << ", Momentum=" << r.momentum
             << ", Accuracy=" << r.avg_accuracy * 100 << "%, Epochs=" << r.avg_epochs << "\n";
    }
}

int main() {
    cout << "Loading dataset cross.csv...\n";
    if (!load_dataset("cross.csv")) {
        cerr << "Failed to load dataset\n";
        return 1;
    }
    cout << "Normalizing dataset features...\n";
    normalize_dataset(dataset);

    // Try several hidden layer configs
    vector<vector<int>> hidden_layer_configs = {
        {3},      // 1 hidden layer with 3 nodes
        {5},      // 1 hidden layer with 5 nodes
        {3, 3},   // 2 hidden layers, 3 nodes each
        {5, 5}    // 2 hidden layers, 5 nodes each
    };

    // Learning rates to test
    vector<double> learning_rates = {0.1, 0.01};

    // Momentum values to test
    vector<double> momentum_values = {0.0, 0.9};

    k_fold_cross_validation(hidden_layer_configs, learning_rates, momentum_values, 2);

    return 0;
}
