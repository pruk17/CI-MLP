#include <iostream>      // input / output
#include <fstream>       // read file
#include <vector>        // vector array 
#include <sstream>       // string -> number
#include <cmath>         // exp(), pow(), etc.
#include <random>        // for random weight
#include <algorithm>     // random_shuffle
#include <cctype>        //functions for checking and manipulating characters.
#include <string>

using namespace std;

// === CONFIG ===
const int INPUT_SIZE = 8;
const int OUTPUT_SIZE = 1;

int NUM_HIDDEN_LAYERS = 2; // Number of hidden layers
vector<int> hidden_sizes = {10, 5}; // Nodes in each hidden layer

//start-up values
double LEARNING_RATE = 0.01;
double MOMENTUM = 0.9;
int EPOCHS = 1000; //test in 1000 loop
int K_FOLD = 10; //for 10% cross validation

enum InitType { BASIC, XAVIER, HE };

string get_init_name(InitType init) {
    switch (init) {
        case BASIC: return "basic ";
        case XAVIER: return "xavier ";
        case HE: return "he ";
    }
    return "unknown";
}


struct Sample {
    vector<double> input;
    vector<double> output;
};

vector<Sample> dataset;

// === Activation Function ===
// use to set the output of each node to be 0 - 1
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}
// use derivative to adjust the weight while training
double sigmoid_derivative(double x) {
    return x * (1.0 - x);
}

// === Shuffle and Normalize ===
    //set the value of out to be in between 0 - 1
void normalize_dataset(vector<Sample>& data) {
    for (int i = 0; i < INPUT_SIZE + OUTPUT_SIZE; ++i) {
        double min_val = 1e9, max_val = -1e9;
        for (const auto& s : data) {
            double val = (i < INPUT_SIZE) ? s.input[i] : s.output[i - INPUT_SIZE];
            min_val = min(min_val, val);
            max_val = max(max_val, val);
        }
        for (auto& s : data) {
            if (i < INPUT_SIZE)
                s.input[i] = (s.input[i] - min_val) / (max_val - min_val);
            else
                s.output[i - INPUT_SIZE] = (s.output[i - INPUT_SIZE] - min_val) / (max_val - min_val);
        }
    }
}

// cut the space, Ex."  95  " (trim)
std::string trim(const std::string& s) {
    size_t start = 0;
    size_t end = s.size();
    //isspace() => check the space, newline
    while (start < end && std::isspace(static_cast<unsigned char>(s[start]))) ++start;
    while (end > start && std::isspace(static_cast<unsigned char>(s[end - 1]))) --end;

    return s.substr(start, end - start);
}

// === Load Dataset ===
    /*read Flood_dataset.csv then skip the first 2 lines
    transfer string ==> double then archive in vector<> with the name "dataset"*/
void load_dataset(string filename) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error: Cannot open file " << filename << endl;
        return;
    }

    string line;

    // ✅ Skip the first 2 header lines
    for (int i = 0; i < 2; ++i) {
        if (!getline(file, line)) {
            cerr << "Error: File doesn't have enough header lines" << endl;
            return;
        }
    }

    while (getline(file, line)) {
        if (line.empty()) continue;

        istringstream ss(line);
        Sample sample;
        string val_str;

        // Read 8 inputs
        for (int i = 0; i < INPUT_SIZE; ++i) {
            if (!getline(ss, val_str, ',')) {
                cerr << "Error: missing input value at line: " << line << endl;
                return;
            }
            try {
                sample.input.push_back(stod(trim(val_str)));
            } catch (const invalid_argument&) {
                cerr << "Invalid number format at line: " << line << endl;
                return;
            }
        }

        // Read 1 output
        if (!getline(ss, val_str, ',')) {
            cerr << "Error: missing output value at line: " << line << endl;
            return;
        }
        try {
            sample.output.push_back(stod(trim(val_str)));
        } catch (const invalid_argument&) {
            cerr << "Invalid number format at line: " << line << endl;
            return;
        }

        dataset.push_back(sample);
    }
}


// === MLP Class ===
/*Multi-Layer Perceptron
    8 input nodes from dataset, 1 output node (water level)
    hidden_size = > hidden node quantity*/
class MLP {
public:
    InitType init_type;
    vector<vector<vector<double>>> weights;   // weights[layer][from][to]
    vector<vector<vector<double>>> delta_weights; // momentum term for weights

    vector<vector<double>> layers; // activations for each layer (including input and output)
    vector<vector<double>> errors; // errors for each layer (excluding input layer)

    MLP(int input_size, const vector<int>& hidden_sizes, int output_size, InitType init_type = BASIC)
        : init_type(init_type) {
        init_network(input_size, hidden_sizes, output_size);
    }

    void init_network(int input_size, const vector<int>& hidden_sizes, int output_size) {
        random_device rd; // Creates a random device to generate random seed, different results each time
        mt19937 gen(rd()); // Initializes a pseudo-random number generator (Mersenne Twister)
                         //, Uses the non-deterministic seed from 'rd' for better randomness

        int prev_size = input_size;
        int total_layers = hidden_sizes.size() + 1;
        weights.resize(total_layers);
        delta_weights.resize(total_layers);

        for (int l = 0; l < total_layers; ++l) {
            int curr_size = (l == total_layers - 1) ? output_size : hidden_sizes[l];
            weights[l].resize(prev_size, vector<double>(curr_size));
            delta_weights[l].resize(prev_size, vector<double>(curr_size));

            for (int i = 0; i < prev_size; ++i) {
                for (int j = 0; j < curr_size; ++j) {
                    double weight = 0;
                    switch (init_type) {
                        case BASIC:
                            weight = ((double)rand() / RAND_MAX) * 0.2 - 0.1; // -0.1 to 0.1
                            break;
                        case XAVIER: {
                            double limit = sqrt(6.0 / (prev_size + curr_size));
                            uniform_real_distribution<> dist(-limit, limit);
                            weight = dist(gen);
                            break;
                        }
                        case HE: {
                            normal_distribution<> dist(0.0, sqrt(2.0 / prev_size));
                            weight = dist(gen);
                            break;
                        }
                    }
                    weights[l][i][j] = weight;
                }
            }

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


    // Forward pass through all layers
    void forward(const vector<double>& input) {
        layers[0] = input;

        for (int l = 0; l < weights.size(); ++l) {
            for (int j = 0; j < layers[l+1].size(); ++j) {
                double sum = 0;
                for (int i = 0; i < layers[l].size(); ++i)
                    sum += layers[l][i] * weights[l][i][j];
                layers[l+1][j] = sigmoid(sum);
            }
        }
    }

    // Backpropagation through all layers
    void backward(const vector<double>& target) {
        int last_layer = errors.size() - 1;

        // Calculate output layer error
        for (int k = 0; k < layers.back().size(); ++k) {
            errors[last_layer][k] = (target[k] - layers.back()[k]) * sigmoid_derivative(layers.back()[k]);
        }

        // Calculate hidden layers errors backward
        for (int l = last_layer - 1; l >= 0; --l) {
            for (int i = 0; i < errors[l].size(); ++i) {
                double err_sum = 0;
                for (int j = 0; j < errors[l+1].size(); ++j)
                    err_sum += errors[l+1][j] * weights[l+1][i][j];
                errors[l][i] = err_sum * sigmoid_derivative(layers[l+1][i]);
            }
        }

        // Update weights with momentum and learning rate
        for (int l = 0; l < weights.size(); ++l) {
            for (int i = 0; i < weights[l].size(); ++i) {
                for (int j = 0; j < weights[l][i].size(); ++j) {
                    double delta = LEARNING_RATE * errors[l][j] * layers[l][i] + MOMENTUM * delta_weights[l][i][j];
                    weights[l][i][j] += delta;
                    delta_weights[l][i][j] = delta;
                }
            }
        }
    }

    // Calculate MSE for output
    double mse(const vector<double>& target) {
        double sum = 0;
        for (int i = 0; i < target.size(); ++i) {
            sum += pow(target[i] - layers.back()[i], 2);
        }
        return sum / target.size();
    }
};

//10% cross validatiom, k = 10
// 9 => train with MLP, 1 => test the accuracy
// k-fold training with varying hyperparameters
void k_fold_train() {
    vector<vector<int>> hidden_layer_options = {
        {10}, {15}, {10, 5}, {15, 10}
    };
    vector<double> learning_rates = {0.01, 0.05, 0.1};
    vector<double> momentum_values = {0.5, 0.9};
    int trial = 1;

    struct ResultSummary {
        vector<int> hidden_layers;
        double lr;
        double momentum;
        double avg_mse;
        InitType init_type;
    };

    vector<ResultSummary> all_results;

    for (int init = 0; init <= 2; ++init) {
        InitType init_type = static_cast<InitType>(init);

        for (auto& hidden_layers : hidden_layer_options) {
            for (double lr : learning_rates) {
                for (double mo : momentum_values) {
                    double trial_total_mse = 0;

                    for (int t = 0; t < trial; ++t) {
                        mt19937 g(random_device{}());
                        shuffle(dataset.begin(), dataset.end(), g);

                        int fold_size = dataset.size() / K_FOLD;
                        double avg_mse = 0;

                        for (int fold = 0; fold < K_FOLD; ++fold) {
                            vector<Sample> train_set, test_set;
                            for (int i = 0; i < dataset.size(); ++i) {
                                if (i >= fold * fold_size && i < (fold + 1) * fold_size)
                                    test_set.push_back(dataset[i]);
                                else
                                    train_set.push_back(dataset[i]);
                            }

                            hidden_sizes = hidden_layers;
                            NUM_HIDDEN_LAYERS = (int)hidden_layers.size();
                            LEARNING_RATE = lr;
                            MOMENTUM = mo;

                            MLP net(INPUT_SIZE, hidden_sizes, OUTPUT_SIZE, init_type);

                            for (int epoch = 0; epoch < EPOCHS; ++epoch) {
                                for (auto& s : train_set) {
                                    net.forward(s.input);
                                    net.backward(s.output);
                                }
                            }

                            double total_error = 0;
                            for (auto& s : test_set) {
                                net.forward(s.input);
                                total_error += net.mse(s.output);
                            }
                            avg_mse += total_error / test_set.size();
                        }

                        double final_avg_mse = avg_mse / K_FOLD;
                        trial_total_mse += final_avg_mse;

                        cout << "[Weight: " << get_init_name(init_type)
                             << "init] [Hidden layers ";
                        for (auto n : hidden_layers) cout << n << " ";
                        cout << ", LR " << lr << ", Momentum " << mo << "] AVG MSE: "
                             << final_avg_mse << "\n";
                    }

                    all_results.push_back({hidden_layers, lr, mo, trial_total_mse / trial, init_type});
                }
            }
        }
    }

    cout << "\n========= BEST RESULT =========\n";

    auto best_result = min_element(all_results.begin(), all_results.end(),
        [](const ResultSummary& a, const ResultSummary& b) {
            return a.avg_mse < b.avg_mse;
        });
        double rmse = sqrt(best_result->avg_mse);
    cout << "[Weight: " << get_init_name(best_result->init_type) << "init] Hidden layers: ";
    for (auto n : best_result->hidden_layers) cout << n << " ";
    cout << ", LR: " << best_result->lr
         << ", Momentum: " << best_result->momentum
         << ", AVG MSE: " << best_result->avg_mse << "| RMSE: " << rmse << "\n";
}


int main() {
    load_dataset("Flood_dataset.csv"); //.txt => .csv for better use in tables
    normalize_dataset(dataset);
    k_fold_train();
    return 0;
}
