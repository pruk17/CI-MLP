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
int HIDDEN_SIZE = 10;

double LEARNING_RATE = 0.01;
double MOMENTUM = 0.9;
int EPOCHS = 1000;
int K_FOLD = 10;

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
    vector<vector<double>> w_input_hidden, w_hidden_output;
    vector<vector<double>> delta_input_hidden, delta_hidden_output;

    vector<double> hidden, output;
    vector<double> error_output, error_hidden;

    MLP(int input_size, int hidden_size, int output_size) {
        init_weights(input_size, hidden_size, output_size);
    }
    //random the first weight
    void init_weights(int input_size, int hidden_size, int output_size) {
        random_device rd;
        mt19937 gen(rd());
        uniform_real_distribution<> dist(-1, 1);

        w_input_hidden.resize(input_size, vector<double>(hidden_size));
        delta_input_hidden.resize(input_size, vector<double>(hidden_size));
        for (auto& row : w_input_hidden)
            for (auto& w : row) w = dist(gen);

        w_hidden_output.resize(hidden_size, vector<double>(output_size));
        delta_hidden_output.resize(hidden_size, vector<double>(output_size));
        for (auto& row : w_hidden_output)
            for (auto& w : row) w = dist(gen);

        hidden.resize(hidden_size);
        output.resize(output_size);
        error_output.resize(output_size);
        error_hidden.resize(hidden_size);
    }
    //calculate output, input => hidden layers => output
    void forward(const vector<double>& input) {
        for (int j = 0; j < hidden.size(); ++j) {
            hidden[j] = 0;
            for (int i = 0; i < input.size(); ++i)
                hidden[j] += input[i] * w_input_hidden[i][j];
            hidden[j] = sigmoid(hidden[j]);
        }
        for (int k = 0; k < output.size(); ++k) {
            output[k] = 0;
            for (int j = 0; j < hidden.size(); ++j)
                output[k] += hidden[j] * w_hidden_output[j][k];
            output[k] = sigmoid(output[k]);
        }
    }
    //weight adjustment with backpropagation
    void backward(const vector<double>& input, const vector<double>& target) {
        for (int k = 0; k < output.size(); ++k)
            error_output[k] = (target[k] - output[k]) * sigmoid_derivative(output[k]);

        for (int j = 0; j < hidden.size(); ++j) {
            error_hidden[j] = 0;
            for (int k = 0; k < output.size(); ++k)
                error_hidden[j] += error_output[k] * w_hidden_output[j][k];
            error_hidden[j] *= sigmoid_derivative(hidden[j]);
        }

        for (int j = 0; j < hidden.size(); ++j) {
            for (int k = 0; k < output.size(); ++k) {
                double delta = LEARNING_RATE * error_output[k] * hidden[j] + MOMENTUM * delta_hidden_output[j][k];
                w_hidden_output[j][k] += delta;
                delta_hidden_output[j][k] = delta;
            }
        }

        for (int i = 0; i < input.size(); ++i) {
            for (int j = 0; j < hidden.size(); ++j) {
                double delta = LEARNING_RATE * error_hidden[j] * input[i] + MOMENTUM * delta_input_hidden[i][j];
                w_input_hidden[i][j] += delta;
                delta_input_hidden[i][j] = delta;
            }
        }
    }
    //find Mean Squared Error from output
    double mse(const vector<double>& target) {
        double sum = 0;
        for (int k = 0; k < target.size(); ++k)
            sum += pow(target[k] - output[k], 2);
        return sum / target.size();
    }
};
//10% cross validatiom, k = 10
// 9 => train with MLP, 1 => test the accuracy
// k-fold training with varying hyperparameters
void k_fold_train() {
    // Define parameters to test
    vector<int> hidden_sizes = {5, 10, 15};
    vector<double> learning_rates = {0.01, 0.05, 0.1};
    vector<double> momentum_values = {0.5, 0.9};
    int trial = 1; // number of different random weight initializations

    struct ResultSummary {
        int hidden;
        double lr;
        double momentum;
        double avg_mse;
    };

    vector<ResultSummary> all_results;

    for (int hs : hidden_sizes) {
        for (double lr : learning_rates) {
            for (double mo : momentum_values) {
                double trial_total_mse = 0;

                for (int t = 1; t <= trial; ++t) {
                    // Shuffle dataset before splitting
                    std::random_device rd;
                    std::mt19937 g(rd());
                    std::shuffle(dataset.begin(), dataset.end(), g);

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

                        // Set global hyperparameters
                        HIDDEN_SIZE = hs;
                        LEARNING_RATE = lr;
                        MOMENTUM = mo;

                        MLP net(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);

                        // === This is where EPOCHS is used ===
                        for (int epoch = 0; epoch < EPOCHS; ++epoch) {
                            for (const auto& s : train_set) {
                                net.forward(s.input);
                                net.backward(s.input, s.output);
                            }
                        }

                        // Calculate error on test set
                        double total_error = 0;
                        for (const auto& s : test_set) {
                            net.forward(s.input);
                            total_error += net.mse(s.output);
                        }

                        avg_mse += total_error / test_set.size(); // accumulate fold mse
                    }

                    double final_avg_mse = avg_mse / K_FOLD;
                    trial_total_mse += final_avg_mse;

                    // Show only final AVG MSE per trial
                    cout << "[Hidden " << hs << ", LR " << lr << ", Momentum " << mo
                         << "] AVG MSE: " << final_avg_mse << "\n";
                }

                // Store result for final summary
                all_results.push_back({hs, lr, mo, trial_total_mse / trial});
            }
        }
    }

    // === Final Summary ===
    cout << "\n========= SUMMARY =========\n";
    for (const auto& r : all_results) {
        cout << "Hidden: " << r.hidden
             << ", LR: " << r.lr
             << ", Momentum: " << r.momentum
             << ", AVG MSE: " << r.avg_mse << "\n";
    }
}

int main() {
    load_dataset("Flood_dataset.csv"); //.txt => .csv for better use in tables
    normalize_dataset(dataset);
    k_fold_train();
    return 0;
}
