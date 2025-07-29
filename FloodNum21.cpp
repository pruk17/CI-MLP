#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <ctime>
#include <algorithm> 
#include <cstdlib>

using namespace std;

// === CONFIG ===
const int INPUT_SIZE = 8;
const int OUTPUT_SIZE = 1;

double LEARNING_RATE = 0.01;
double MOMENTUM = 0.9;
int EPOCHS = 1000;
int K_FOLD = 10;
int TRIALS = 3;  // จำนวนครั้งที่ initialize weight ใหม่ต่อ config

struct Sample {
    vector<double> input;
    vector<double> output;
};

vector<Sample> dataset;

// === Activation Functions ===
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}
double sigmoid_derivative(double x) {
    return x * (1.0 - x);
}

// === Normalize Dataset ===
void normalize_dataset(vector<Sample>& data) {
    for (int i = 0; i < INPUT_SIZE + OUTPUT_SIZE; ++i) {
        double min_val = 1e9, max_val = -1e9;
        for (const auto& s : data) {
            double val = (i < INPUT_SIZE) ? s.input[i] : s.output[i - INPUT_SIZE];
            if (val < min_val) min_val = val;
            if (val > max_val) max_val = val;
        }
        for (auto& s : data) {
            if (i < INPUT_SIZE)
                s.input[i] = (s.input[i] - min_val) / (max_val - min_val);
            else
                s.output[i - INPUT_SIZE] = (s.output[i - INPUT_SIZE] - min_val) / (max_val - min_val);
        }
    }
}

// === Load Dataset (manual CSV parser) ===
void load_dataset(const string& filename) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error: Cannot open file " << filename << endl;
        exit(1);
    }

    string line;
    for (int i = 0; i < 2; ++i) getline(file, line); // skip 2 header lines

    while (getline(file, line)) {
        if (line.empty()) continue;

        Sample sample;
        string value = "";
        int col = 0;
        for (int i = 0; i <= (int)line.size(); ++i) {
            if (i == (int)line.size() || line[i] == ',') {
                if (!value.empty()) {
                    double val = atof(value.c_str());
                    if (col < INPUT_SIZE)
                        sample.input.push_back(val);
                    else
                        sample.output.push_back(val);
                    value = "";
                    col++;
                }
            } else {
                value += line[i];
            }
        }

        if (sample.input.size() == INPUT_SIZE && sample.output.size() == OUTPUT_SIZE)
            dataset.push_back(sample);
    }
}

// === MLP Class ===
class MLP {
public:
    vector<vector<vector<double>>> weights;    // [layer][from][to]
    vector<vector<vector<double>>> delta_weights; // for momentum
    vector<vector<double>> layers;              // activations
    vector<vector<double>> errors;              // error terms (no input layer)

    MLP(int input_size, const vector<int>& hidden_sizes, int output_size) {
        init_network(input_size, hidden_sizes, output_size);
    }

    void init_network(int input_size, const vector<int>& hidden_sizes, int output_size) {
        srand((unsigned)time(0));

        int prev_size = input_size;
        int total_layers = (int)hidden_sizes.size() + 1; // hidden + output

        weights.resize(total_layers);
        delta_weights.resize(total_layers);

        for (int l = 0; l < total_layers; ++l) {
            int curr_size = (l == total_layers - 1) ? output_size : hidden_sizes[l];
            weights[l].resize(prev_size, vector<double>(curr_size));
            delta_weights[l].resize(prev_size, vector<double>(curr_size, 0.0));
            for (int i = 0; i < prev_size; ++i)
                for (int j = 0; j < curr_size; ++j)
                    weights[l][i][j] = ((double)rand() / RAND_MAX) * 2 - 1; // random between -1 and 1
            prev_size = curr_size;
        }

        layers.resize(total_layers + 1);
        layers[0].resize(input_size);
        for (int i = 0; i < (int)hidden_sizes.size(); ++i)
            layers[i + 1].resize(hidden_sizes[i]);
        layers[total_layers].resize(output_size);

        errors.resize(total_layers);
        for (int i = 0; i < total_layers; ++i) {
            int sz = (i == total_layers - 1) ? output_size : hidden_sizes[i];
            errors[i].resize(sz);
        }
    }

    void forward(const vector<double>& input) {
        layers[0] = input;
        for (int l = 0; l < (int)weights.size(); ++l) {
            for (int j = 0; j < (int)layers[l + 1].size(); ++j) {
                double sum = 0;
                for (int i = 0; i < (int)layers[l].size(); ++i) {
                    sum += layers[l][i] * weights[l][i][j];
                }
                layers[l + 1][j] = sigmoid(sum);
            }
        }
    }

    void backward(const vector<double>& target) {
        int last_layer = (int)errors.size() - 1;

        // output layer error
        for (int k = 0; k < (int)layers.back().size(); ++k) {
            errors[last_layer][k] = (target[k] - layers.back()[k]) * sigmoid_derivative(layers.back()[k]);
        }

        // hidden layers error backpropagation
        for (int l = last_layer - 1; l >= 0; --l) {
            for (int i = 0; i < (int)errors[l].size(); ++i) {
                double err_sum = 0;
                for (int j = 0; j < (int)errors[l + 1].size(); ++j) {
                    err_sum += errors[l + 1][j] * weights[l + 1][i][j];
                }
                errors[l][i] = err_sum * sigmoid_derivative(layers[l + 1][i]);
            }
        }

        // update weights with learning rate and momentum
        for (int l = 0; l < (int)weights.size(); ++l) {
            for (int i = 0; i < (int)weights[l].size(); ++i) {
                for (int j = 0; j < (int)weights[l][i].size(); ++j) {
                    double delta = LEARNING_RATE * errors[l][j] * layers[l][i] + MOMENTUM * delta_weights[l][i][j];
                    weights[l][i][j] += delta;
                    delta_weights[l][i][j] = delta;
                }
            }
        }
    }

    double mse(const vector<double>& target) {
        double sum = 0;
        for (int i = 0; i < (int)target.size(); ++i) {
            sum += pow(target[i] - layers.back()[i], 2);
        }
        return sum / target.size();
    }
};

// === k-fold cross validation + hyperparameter tuning ===
void k_fold_train() {
    // Hyperparameter options to test
    vector<vector<int>> hidden_layer_options = {
        {10}, {15}, {10, 5}, {15, 10}
    };
    vector<double> learning_rates = {0.01, 0.05, 0.1};
    vector<double> momentum_values = {0.5, 0.9};

    struct Result {
        vector<int> hidden_layers;
        double lr;
        double momentum;
        double avg_mse;
    };

    vector<Result> all_results;

    for (auto& hidden_layers : hidden_layer_options) {
        for (double lr : learning_rates) {
            for (double mo : momentum_values) {
                LEARNING_RATE = lr;
                MOMENTUM = mo;

                double total_trial_mse = 0;

                for (int trial = 0; trial < TRIALS; ++trial) {
                    // Shuffle dataset for each trial with different seed
                    srand((unsigned)time(0) + trial * 100);
                    random_shuffle(dataset.begin(), dataset.end());

                    double fold_mse_sum = 0;
                    int fold_size = (int)dataset.size() / K_FOLD;

                    for (int fold = 0; fold < K_FOLD; ++fold) {
                        vector<Sample> train_set, test_set;

                        for (int i = 0; i < (int)dataset.size(); ++i) {
                            if (i >= fold * fold_size && i < (fold + 1) * fold_size)
                                test_set.push_back(dataset[i]);
                            else
                                train_set.push_back(dataset[i]);
                        }

                        MLP net(INPUT_SIZE, hidden_layers, OUTPUT_SIZE);

                        for (int epoch = 0; epoch < EPOCHS; ++epoch) {
                            for (auto& s : train_set) {
                                net.forward(s.input);
                                net.backward(s.output);
                            }
                        }

                        double fold_error = 0;
                        for (auto& s : test_set) {
                            net.forward(s.input);
                            fold_error += net.mse(s.output);
                        }
                        fold_mse_sum += fold_error / test_set.size();
                    }

                    total_trial_mse += fold_mse_sum / K_FOLD;
                }

                double avg_mse = total_trial_mse / TRIALS;
                all_results.push_back({hidden_layers, lr, mo, avg_mse});

                cout << "[Hidden layers ";
                for (auto n : hidden_layers) cout << n << " ";
                cout << ", LR " << lr << ", Momentum " << mo << "] AVG MSE: " << avg_mse << "\n";
            }
        }
    }

    // Find best config (lowest avg MSE)
    auto best = all_results.begin();
    for (auto it = all_results.begin(); it != all_results.end(); ++it) {
        if (it->avg_mse < best->avg_mse)
            best = it;
    }

    cout << "\n===== BEST CONFIGURATION =====\n";
    cout << "Hidden layers: ";
    for (auto n : best->hidden_layers) cout << n << " ";
    cout << "\nLearning Rate: " << best->lr << "\nMomentum: " << best->momentum << "\nAverage MSE: " << best->avg_mse << "\n";
}

int main() {
    load_dataset("Flood_dataset.csv");
    normalize_dataset(dataset);
    k_fold_train();
    return 0;
}
