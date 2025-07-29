#include <iostream>      // input / output
#include <fstream>       // read file
#include <vector>        // vector array 
#include <sstream>       // string -> number
#include <cmath>         // exp(), pow(), etc.
#include <random>        // for random weight
#include <algorithm>     // random_shuffle

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
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}
double sigmoid_derivative(double x) {
    return x * (1.0 - x);
}

// === Shuffle and Normalize ===
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

// === Load Dataset ===
void load_dataset(string filename) {
    ifstream file(filename);
    string line;
    getline(file, line); // skip header
    
    while (getline(file, line)) {
        istringstream ss(line);
        Sample sample;
        string val_str;
        
        // read 8 output
        for (int i = 0; i < INPUT_SIZE; ++i) {
            if (!getline(ss, val_str, ',')) {
                cerr << "Error: missing input value at line: " << line << endl;
                return;
            }
            sample.input.push_back(stod(val_str));
        }
        
        // read one output value
        if (!getline(ss, val_str, ',')) {
            cerr << "Error: missing output value at line: " << line << endl;
            return;
        }
        sample.output.push_back(stod(val_str));
        
        dataset.push_back(sample);
    }
}

// === MLP Class ===
class MLP {
public:
    vector<vector<double>> w_input_hidden, w_hidden_output;
    vector<vector<double>> delta_input_hidden, delta_hidden_output;

    vector<double> hidden, output;
    vector<double> error_output, error_hidden;

    MLP(int input_size, int hidden_size, int output_size) {
        init_weights(input_size, hidden_size, output_size);
    }

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

    double mse(const vector<double>& target) {
        double sum = 0;
        for (int k = 0; k < target.size(); ++k)
            sum += pow(target[k] - output[k], 2);
        return sum / target.size();
    }
};

void k_fold_train() {
    
    std::random_device rd; // Creates a random device to generate random seed, different results each time
    std::mt19937 g(rd()); // Initializes a pseudo-random number generator (Mersenne Twister)
                         //, Uses the non-deterministic seed from 'rd' for better randomness
    
    std::shuffle(dataset.begin(), dataset.end(), g); // Shuffles the order of all sample
                         // Uses the random number generator 'g' to ensure proper, unbiased shuffling
    int fold_size = dataset.size() / K_FOLD;
    for (int fold = 0; fold < K_FOLD; ++fold) {
        vector<Sample> train_set, test_set;
        for (int i = 0; i < dataset.size(); ++i) {
            if (i >= fold * fold_size && i < (fold + 1) * fold_size)
                test_set.push_back(dataset[i]);
            else
                train_set.push_back(dataset[i]);
        }

        MLP net(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);
        for (int epoch = 0; epoch < EPOCHS; ++epoch) {
            for (const auto& s : train_set) {
                net.forward(s.input);
                net.backward(s.input, s.output);
            }
        }

        double total_error = 0;
        for (const auto& s : test_set) {
            net.forward(s.input);
            total_error += net.mse(s.output);
        }
        cout << "Fold " << fold + 1 << " MSE: " << total_error / test_set.size() << endl;
        // MSE => Mean Squared Error, ค่าความผิดพลาดเฉลี่ย
        //Mean squared error between the actual (target) value and the MLP predicted value.
    }
}

int main() {
    load_dataset("Flood_dataset.csv"); //.txt => .csv for better use in tables
    normalize_dataset(dataset);
    k_fold_train();
    return 0;
}
