#include <iostream> // input / output
#include <vector>   // vector array 
#include <fstream>  // read file
#include <sstream>  // string -> number
#include <cmath>    // exp(), pow(), etc.
//#include <algorithm>
#include <random> 
using namespace std;

enum InitType { BASIC, XAVIER, HE };

// === CONFIG ===
const int INPUT_SIZE = 2;
const int OUTPUT_SIZE = 1;
int EPOCHS = 1000;
int K_FOLD = 10;

vector<vector<int>> hidden_layer_options = {
    {10}, {15}, {10, 5}, {15, 10}
};
vector<double> learning_rates = {0.01, 0.05, 0.1};
vector<double> momentum_values = {0.5, 0.9};
vector<InitType> init_methods = {BASIC, XAVIER, HE};

string get_init_name(InitType init) { //init => initialization 
    switch (init) {
        case BASIC: return "basic "; //random from constant values : -0.1. 0.1
        case XAVIER: return "xavier "; //from formula : Xavier Glorot Initialization
        case HE: return "he "; //from formula : He-et-al Initialization
    }
    return "unknown";
}

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
    random_device rd; // Creates a random device to generate random seed, different results each time
    mt19937 g(rd()); // Initializes a pseudo-random number generator (Mersenne Twister)
                             //, Uses the non-deterministic seed from 'rd' for better randomness
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
    InitType init_type;

    MLP(vector<int> layer_config, double lr, double mmt, InitType init)
        : layers(layer_config), LEARNING_RATE(lr), MOMENTUM(mmt), init_type(init) {
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

        random_device rd;
        mt19937 gen(rd());
        uniform_real_distribution<double> u(-1, 1);
        normal_distribution<double> n(0.0, 1.0);

        for (size_t i = 0; i < weights.size(); ++i) {
            int fan_in = layers[i];
            int fan_out = layers[i + 1];

            weights[i].resize(fan_out);
            prev_deltas[i].resize(fan_out);

            for (int j = 0; j < fan_out; ++j) {
                weights[i][j].resize(fan_in + 1);
                prev_deltas[i][j].resize(fan_in + 1);
                for (int k = 0; k <= fan_in; ++k) {
                    double w = 0;
                    if (init_type == BASIC) {
                        w = u(gen) * 0.1;
                    } else if (init_type == XAVIER) {
                        double limit = sqrt(6.0 / (fan_in + fan_out));
                        w = u(gen) * limit;
                    } else if (init_type == HE) {
                        double stddev = sqrt(2.0 / fan_in);
                        w = n(gen) * stddev;
                    }
                    weights[i][j][k] = w;
                    prev_deltas[i][j][k] = 0.0;
                }
            }
        }
    }

    vector<double> forward(const vector<double>& input) {
        neurons[0] = input;
        for (size_t i = 1; i < layers.size(); ++i) {
            for (int j = 0; j < layers[i]; ++j) {
                double sum = weights[i - 1][j][layers[i - 1]];
                for (int k = 0; k < layers[i - 1]; ++k)
                    sum += weights[i - 1][j][k] * neurons[i - 1][k];
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
                for (int k = 0; k < layers[i + 1]; ++k)
                    sum += weights[i][k][j] * deltas[i + 1][k];
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

    int train(const vector<Sample>& data, double mse_threshold) {
        int converged_epoch = EPOCHS;
        for (int epoch = 0; epoch < EPOCHS; ++epoch) {
            double total_mse = 0.0;
            for (auto& sample : data) {
                forward(sample.input);
                total_mse += backward(sample.target);
            }
            if (total_mse / data.size() < mse_threshold) {
                converged_epoch = epoch + 1;
                break;
            }
        }
        return converged_epoch;
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
        if (line.empty() || line[0] == 'p') continue;
        stringstream ss(line);
        Sample s;
        double a, b;
        ss >> a; if (ss.peek() == ',') ss.ignore();
        ss >> b;
        s.input = {a, b};
        if (!getline(file, line)) break;
        stringstream st(line);
        int t;
        st >> t;
        s.target = {double(t)};
        dataset.push_back(s);
    }
    return dataset;
}


void cross_validate(vector<Sample>& dataset, double mse_threshold) {
    double best_acc = -1.0;
    vector<int> best_hidden;
    double best_lr = 0.0, best_mmt = 0.0;
    InitType best_init = BASIC;
    int best_avg_epochs = 0;
    int best_TP = 0, best_FP = 0, best_TN = 0, best_FN = 0;

    for (auto& init : init_methods) {
        int total_TP = 0, total_FP = 0, total_TN = 0, total_FN = 0;

        for (auto& hidden_config : hidden_layer_options) {
            for (auto& lr : learning_rates) {
                for (auto& mmt : momentum_values) {
                    int fold_size = dataset.size() / K_FOLD;
                    shuffle_data(dataset);
                    int TP = 0, FP = 0, TN = 0, FN = 0;
                    int total_epochs = 0;

                    for (int k = 0; k < K_FOLD; ++k) {
                        vector<Sample> train_data, test_data;
                        for (int i = 0; i < dataset.size(); ++i) {
                            if (i >= k * fold_size && i < (k + 1) * fold_size)
                                test_data.push_back(dataset[i]);
                            else
                                train_data.push_back(dataset[i]);
                        }
                        vector<int> layer_config = {INPUT_SIZE};
                        layer_config.insert(layer_config.end(), hidden_config.begin(), hidden_config.end());
                        layer_config.push_back(OUTPUT_SIZE);

                        MLP mlp(layer_config, lr, mmt, init);
                        int used_epochs = mlp.train(train_data, mse_threshold);
                        total_epochs += used_epochs;

                        for (auto& sample : test_data) {
                            int pred = mlp.predict(sample.input);
                            int actual = static_cast<int>(sample.target[0]);
                            if (pred == 1 && actual == 1) TP++;
                            else if (pred == 1 && actual == 0) FP++;
                            else if (pred == 0 && actual == 0) TN++;
                            else if (pred == 0 && actual == 1) FN++;
                        }
                    }

                    total_TP += TP;
                    total_FP += FP;
                    total_TN += TN;
                    total_FN += FN;

                    double acc = 100.0 * (TP + TN) / (TP + TN + FP + FN);
                    int avg_epochs = total_epochs / K_FOLD;

                    cout << "[Weight: " << get_init_name(init) << "] [Hidden layers ";
                    for (int h : hidden_config) cout << h << " ";
                    cout << ", LR " << lr << ", Momentum " << mmt << "] Accuracy: " << acc
                         << "%, Epochs to converge: " << avg_epochs << "\n";

                    if (acc > best_acc) {
                        best_acc = acc;
                        best_hidden = hidden_config;
                        best_lr = lr;
                        best_mmt = mmt;
                        best_init = init;
                        best_avg_epochs = avg_epochs;
                        best_TP = TP;
                        best_FP = FP;
                        best_TN = TN;
                        best_FN = FN;
                    }
                }
            }
        }

        // Confusion Matrix Summary for this activation
        cout << "======= Confusion Matrix for " << get_init_name(init) << " =======\n";
        cout << "| TruePositive : " << total_TP << "  | FalseNegative : " << total_FN << " |\n";
        cout << "| FalsePositive: " << total_FP << "  | TrueNegative : " << total_TN << " |\n\n";
    }

    cout << "========= BEST RESULT =========\n";
    cout << "Train with: Cross.pat => Cross.csv\n";
    cout << "[" << get_init_name(best_init) << "] [Hidden layers ";
    for (int h : best_hidden) cout << h << " ";
    cout << ", LR " << best_lr << ", Momentum " << best_mmt
         << "] Accuracy: " << best_acc << "%, Epochs to converge: " << best_avg_epochs << "\n";

    cout << "======= Confusion Matrix (Best) =======\n";
    cout << "| TruePositive : " << best_TP << "  | FalseNegative : " << best_FN << " |\n";
    cout << "| FalsePositive: " << best_FP << "  | TrueNegative : " << best_TN << " |\n";
}



int main() {
    string filename = "cross.csv";
    vector<Sample> data = load_csv(filename);
    if (data.empty()) {
        cerr << "Failed to load dataset.\n";
        return 1;
    }
    //after trained 0.01, 0.02, 0.05 : 0.02 gives the best precise
    double threshold = 0.02; //value for early stopping
    cout << "\n MSE Threshold = " << threshold << "\n";
    cross_validate(data, threshold);

    return 0;
}

