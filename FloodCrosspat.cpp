#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <cmath>
#include <random>
#include <iomanip>

using namespace std;

// Activation function (sigmoid) และอนุพันธ์
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}
double sigmoid_derivative(double y) {
    return y * (1.0 - y);  // y = sigmoid(x)
}

// โหลด dataset จากไฟล์
void load_dataset(const string& filename, vector<vector<double>>& inputs, vector<vector<double>>& outputs) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error opening file: " << filename << endl;
        exit(1);
    }

    string line;
    while (getline(file, line)) {
        string label = line;
        if (label.empty()) continue;

        // อ่าน input
        if (!getline(file, line)) {
            cerr << "Unexpected EOF after label " << label << endl;
            exit(1);
        }
        istringstream ss_input(line);
        vector<double> input_values;
        string token;
        while (getline(ss_input, token, ',')) {
            try {
                input_values.push_back(stod(token));
            } catch (...) {
                cerr << "Invalid input value after label " << label << endl;
                exit(1);
            }
        }
        if (input_values.size() != 2) {
            cerr << "Input size != 2 after label " << label << endl;
            exit(1);
        }

        // อ่าน output
        if (!getline(file, line)) {
            cerr << "Unexpected EOF after input for label " << label << endl;
            exit(1);
        }
        istringstream ss_output(line);
        vector<double> output_values;
        while (getline(ss_output, token, ',')) {
            try {
                output_values.push_back(stod(token));
            } catch (...) {
                cerr << "Invalid output value after label " << label << endl;
                exit(1);
            }
        }
        if (output_values.size() != 2) {
            cerr << "Output size != 2 after label " << label << endl;
            exit(1);
        }

        inputs.push_back(input_values);
        outputs.push_back(output_values);
    }

    file.close();
}

// MLP Class
class MLP {
public:
    int input_size;
    int hidden_size;
    int output_size;
    double learning_rate;
    double momentum;

    vector<vector<double>> w_input_hidden;   // weights input->hidden
    vector<double> bias_hidden;
    vector<vector<double>> w_hidden_output;  // weights hidden->output
    vector<double> bias_output;

    vector<double> hidden_layer_output;
    vector<double> output_layer_output;

    vector<vector<double>> delta_w_input_hidden;  // สำหรับ momentum
    vector<double> delta_bias_hidden;
    vector<vector<double>> delta_w_hidden_output;
    vector<double> delta_bias_output;

    MLP(int input_size, int hidden_size, int output_size,
        double learning_rate = 0.01, double momentum = 0.5)
        : input_size(input_size), hidden_size(hidden_size), output_size(output_size),
          learning_rate(learning_rate), momentum(momentum) {
        init_weights();
    }

    void init_weights() {
        random_device rd;
        mt19937 gen(rd());
        uniform_real_distribution<> dis(-1.0, 1.0);

        w_input_hidden.resize(hidden_size, vector<double>(input_size));
        bias_hidden.resize(hidden_size);
        w_hidden_output.resize(output_size, vector<double>(hidden_size));
        bias_output.resize(output_size);

        delta_w_input_hidden.resize(hidden_size, vector<double>(input_size, 0.0));
        delta_bias_hidden.resize(hidden_size, 0.0);
        delta_w_hidden_output.resize(output_size, vector<double>(hidden_size, 0.0));
        delta_bias_output.resize(output_size, 0.0);

        for (int i = 0; i < hidden_size; ++i) {
            for (int j = 0; j < input_size; ++j) {
                w_input_hidden[i][j] = dis(gen);
            }
            bias_hidden[i] = dis(gen);
        }

        for (int i = 0; i < output_size; ++i) {
            for (int j = 0; j < hidden_size; ++j) {
                w_hidden_output[i][j] = dis(gen);
            }
            bias_output[i] = dis(gen);
        }
    }

    vector<double> forward(const vector<double>& input) {
        hidden_layer_output.resize(hidden_size);
        output_layer_output.resize(output_size);

        // input -> hidden
        for (int i = 0; i < hidden_size; ++i) {
            double sum = bias_hidden[i];
            for (int j = 0; j < input_size; ++j) {
                sum += w_input_hidden[i][j] * input[j];
            }
            hidden_layer_output[i] = sigmoid(sum);
        }

        // hidden -> output
        for (int i = 0; i < output_size; ++i) {
            double sum = bias_output[i];
            for (int j = 0; j < hidden_size; ++j) {
                sum += w_hidden_output[i][j] * hidden_layer_output[j];
            }
            output_layer_output[i] = sigmoid(sum);
        }

        return output_layer_output;
    }

    void train(const vector<double>& input, const vector<double>& target) {
        forward(input);

        // คำนวณ error output layer
        vector<double> output_errors(output_size);
        for (int i = 0; i < output_size; ++i) {
            output_errors[i] = (target[i] - output_layer_output[i]) * sigmoid_derivative(output_layer_output[i]);
        }

        // คำนวณ error hidden layer
        vector<double> hidden_errors(hidden_size, 0.0);
        for (int i = 0; i < hidden_size; ++i) {
            double error = 0.0;
            for (int j = 0; j < output_size; ++j) {
                error += output_errors[j] * w_hidden_output[j][i];
            }
            hidden_errors[i] = error * sigmoid_derivative(hidden_layer_output[i]);
        }

        // ปรับ weight hidden -> output
        for (int i = 0; i < output_size; ++i) {
            for (int j = 0; j < hidden_size; ++j) {
                double delta = learning_rate * output_errors[i] * hidden_layer_output[j] + momentum * delta_w_hidden_output[i][j];
                w_hidden_output[i][j] += delta;
                delta_w_hidden_output[i][j] = delta;
            }
            double delta_b = learning_rate * output_errors[i] + momentum * delta_bias_output[i];
            bias_output[i] += delta_b;
            delta_bias_output[i] = delta_b;
        }

        // ปรับ weight input -> hidden
        for (int i = 0; i < hidden_size; ++i) {
            for (int j = 0; j < input_size; ++j) {
                double delta = learning_rate * hidden_errors[i] * input[j] + momentum * delta_w_input_hidden[i][j];
                w_input_hidden[i][j] += delta;
                delta_w_input_hidden[i][j] = delta;
            }
            double delta_b = learning_rate * hidden_errors[i] + momentum * delta_bias_hidden[i];
            bias_hidden[i] += delta_b;
            delta_bias_hidden[i] = delta_b;
        }
    }

    double calc_mse(const vector<vector<double>>& inputs, const vector<vector<double>>& targets) {
        double mse = 0.0;
        int n = inputs.size();
        for (int i = 0; i < n; ++i) {
            vector<double> out = forward(inputs[i]);
            for (int j = 0; j < output_size; ++j) {
                double e = targets[i][j] - out[j];
                mse += e * e;
            }
        }
        return mse / n;
    }
};

int main() {
    vector<vector<double>> inputs;
    vector<vector<double>> outputs;

    load_dataset("cross.csv", inputs, outputs);

    int hidden_nodes = 5;       // ปรับได้
    double learning_rate = 0.01;  // ปรับได้
    double momentum = 0.5;        // ปรับได้
    int epochs = 1000;

    MLP mlp(2, hidden_nodes, 2, learning_rate, momentum);

    for (int epoch = 1; epoch <= epochs; ++epoch) {
        for (size_t i = 0; i < inputs.size(); ++i) {
            mlp.train(inputs[i], outputs[i]);
        }

        if (epoch % 100 == 0) {
            double mse = mlp.calc_mse(inputs, outputs);
            cout << fixed << setprecision(7);
            cout << "[Hidden layers " << hidden_nodes << " , LR " << learning_rate << ", Momentum " << momentum << "] "
                 << "Epoch " << epoch << " AVG MSE: " << mse << endl;
        }
    }

    return 0;
}
