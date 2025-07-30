#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <random>
using namespace std;

// === CONFIG ===
const int INPUT_SIZE = 2;
const int OUTPUT_SIZE = 1;

int EPOCHS = 1000;
int K_FOLD = 10;

vector<vector<int>> hidden_layer_options = {
    {10},      // 1 hidden layer with 10 nodes
    {15},      // 1 hidden layer with 15 nodes
    {10, 5},   // 2 hidden layers with 10 and 5 nodes
    {15, 10}   // 2 hidden layers with 15 and 10 nodes
};
vector<double> learning_rates = {0.01, 0.05, 0.1};
vector<double> momentum_values = {0.5, 0.9};

struct Sample {
    vector<double> input;
    vector<double> target;
};

double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double sigmoid_derivative(double y) {
    return y * (1.0 - y);
}

void shuffle_data(vector<Sample>& data) {
    random_device rd;
    mt19937 g(rd());
    shuffle(data.begin(), data.end(), g);
}

class MLP {
public:
    vector<int> layers;
    vector<vector<vector<double>>> weights;
    vector<vector<vector<double>>> prev_deltas;
    vector<vector<double>> neurons;
    vector<vector<double>> deltas;
    double LEARNING_RATE;
    double MOMENTUM;

    MLP(vector<int> layer_config, double lr, double mmt) : layers(layer_config), LEARNING_RATE(lr), MOMENTUM(mmt) {
        init_network();
    }

    void init_network() {
        neurons.resize(layers.size());
        deltas.resize(layers.size());
        weights.resize(layers.size() - 1);
        prev_deltas.resize(layers.size() - 1);

        for (size_t i = 0; i < layers.size(); ++i) {
            neurons[i].resize(layers[i], 0.0);
            deltas[i].resize(layers[i], 0.0);
        }

        for (size_t i = 0; i < weights.size(); ++i) {
            weights[i].resize(layers[i + 1]);
            prev_deltas[i].resize(layers[i + 1]);
            for (int j = 0; j < layers[i + 1]; ++j) {
                weights[i][j].resize(layers[i] + 1);
                prev_deltas[i][j].resize(layers[i] + 1);
                for (int k = 0; k <= layers[i]; ++k) {
                    weights[i][j][k] = 0.01 * (i + j + k + 1);
                    prev_deltas[i][j][k] = 0.0;
                }
            }
        }
    }

    vector<double> forward(const vector<double>& input) {
        neurons[0] = input;
        for (size_t i = 1; i < layers.size(); ++i) {
            for (int j = 0; j < layers[i]; ++j) {
                double sum = weights[i - 1][j][layers[i - 1]]; // bias
                for (int k = 0; k < layers[i - 1]; ++k) {
                    sum += weights[i - 1][j][k] * neurons[i - 1][k];
                }
                neurons[i][j] = sigmoid(sum);
            }
        }
        return neurons.back();
    }

    double backward(const vector<double>& target) {
        double mse = 0.0;
        size_t last = layers.size() - 1;
        for (int i = 0; i < layers[last]; ++i) {
            double error = target[i] - neurons[last][i];
            deltas[last][i] = error * sigmoid_derivative(neurons[last][i]);
            mse += error * error;
        }

        for (int i = layers.size() - 2; i > 0; --i) {
            for (int j = 0; j < layers[i]; ++j) {
                double sum = 0.0;
                for (int k = 0; k < layers[i + 1]; ++k) {
                    sum += weights[i][k][j] * deltas[i + 1][k];
                }
                deltas[i][j] = sum * sigmoid_derivative(neurons[i][j]);
            }
        }

        for (size_t i = 0; i < weights.size(); ++i) {
            for (int j = 0; j < layers[i + 1]; ++j) {
                for (int k = 0; k < layers[i]; ++k) {
                    double delta = LEARNING_RATE * deltas[i + 1][j] * neurons[i][k] + MOMENTUM * prev_deltas[i][j][k];
                    weights[i][j][k] += delta;
                    prev_deltas[i][j][k] = delta;
                }
                double delta = LEARNING_RATE * deltas[i + 1][j] * 1.0 + MOMENTUM * prev_deltas[i][j][layers[i]];
                weights[i][j][layers[i]] += delta;
                prev_deltas[i][j][layers[i]] = delta;
            }
        }

        return mse / layers[last];
    }

    void train(const vector<Sample>& data) {
        for (int epoch = 0; epoch < EPOCHS; ++epoch) {
            double total_mse = 0.0;
            for (auto& sample : data) {
                forward(sample.input);
                total_mse += backward(sample.target);
            }
            if (total_mse / data.size() < 0.001) break;
        }
    }

    int predict(const vector<double>& input) {
        double output = forward(input)[0];
        return output >= 0.5 ? 1 : 0;
    }
};

vector<Sample> load_csv(const string& filename) {
    vector<Sample> dataset;
    ifstream file(filename);
    string line;
    while (getline(file, line)) {
        if (line.empty() || line[0] == 'p') continue; // skip header or empty
        stringstream ss(line);
        Sample s;
        double a, b;
        ss >> a;
        if (ss.peek() == ',') ss.ignore();
        ss >> b;
        s.input = {a, b};
        if (!getline(file, line)) break;
        stringstream st(line);
        int t1;
        st >> t1;
        s.target = {double(t1)};
        dataset.push_back(s);
    }
    return dataset;
}

void cross_validate_trials(vector<Sample>& dataset) {
    double best_acc = -1.0;
    vector<int> best_hidden;
    double best_lr = 0.0, best_mmt = 0.0;

    for (auto& hidden_config : hidden_layer_options) {
        for (auto& lr : learning_rates) {
            for (auto& mmt : momentum_values) {
                int fold_size = dataset.size() / K_FOLD;
                shuffle_data(dataset);
                int TP = 0, FP = 0, TN = 0, FN = 0;
                for (int k = 0; k < K_FOLD; ++k) {
                    vector<Sample> train_data, test_data;
                    for (int i = 0; i < dataset.size(); ++i) {
                        if (i >= k * fold_size && i < (k + 1) * fold_size)
                            test_data.push_back(dataset[i]);
                        else
                            train_data.push_back(dataset[i]);
                    }
                    vector<int> layer_config;
                    layer_config.push_back(INPUT_SIZE);
                    for (int h : hidden_config) layer_config.push_back(h);
                    layer_config.push_back(OUTPUT_SIZE);

                    MLP mlp(layer_config, lr, mmt);
                    mlp.train(train_data);

                    for (auto& sample : test_data) {
                        int pred = mlp.predict(sample.input);
                        int actual = static_cast<int>(sample.target[0]);
                        if (pred == 1 && actual == 1) TP++;
                        else if (pred == 1 && actual == 0) FP++;
                        else if (pred == 0 && actual == 0) TN++;
                        else if (pred == 0 && actual == 1) FN++;
                    }
                }
                double acc = 100.0 * (TP + TN) / (TP + TN + FP + FN);

                // แสดงผลแบบใหม่
                cout << "[Hidden layers ";
                for (int h : hidden_config) cout << h << " ";
                cout << ", LR " << lr << ", Momentum " << mmt << "] Accuracy: " << acc << "%\n";

                if (acc > best_acc) {
                    best_acc = acc;
                    best_hidden = hidden_config;
                    best_lr = lr;
                    best_mmt = mmt;
                }
            }
        }
    }

    cout << "========= BEST RESULT =========\n";
    cout << "Train with: Cross.pat => Cross.csv\n";
    cout << "[Hidden layers ";
    for (int h : best_hidden) cout << h << " ";
    cout << ", LR " << best_lr << ", Momentum " << best_mmt << "] Accuracy: " << best_acc << "%\n";
}

int main() {
    string filename = "cross.csv";
    vector<Sample> data = load_csv(filename);
    if (data.empty()) {
        cerr << "Failed to load dataset.\n";
        return 1;
    }
    cross_validate_trials(data);
    return 0;
}
