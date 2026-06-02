#include "Layer.h"
#include <cmath>
#include <random>
#include <algorithm>

Layer::Layer(int input_size, int output_size, ActivationType activation)
        : input_size(input_size),
            output_size(output_size),
            activation(activation) {
    
    // Initialize weights and biases
    weights.resize(output_size, std::vector<double>(input_size));
    biases.resize(output_size);
    weight_grads.assign(output_size, std::vector<double>(input_size, 0.0));
    bias_grads.assign(output_size, 0.0);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    
    // He initialization (recommended for ReLU)
    std::normal_distribution<> d(0, std::sqrt(2.0 / input_size));
    
    for (int i = 0; i < output_size; ++i) {
        biases[i] = 0.0; // Initialize biases to 0
        for (int j = 0; j < input_size; ++j) {
            weights[i][j] = d(gen);
        }
    }
}

double Layer::relu(double x) {
    return x > 0.0 ? x : 0.0;
}

double Layer::relu_derivative(double x) {
    return x > 0.0 ? 1.0 : 0.0;
}

double Layer::activate(double x) {
    switch (activation) {
        case ActivationType::Relu:
            return relu(x);
        default:
            return relu(x);
    }
}

double Layer::activation_derivative(double x) {
    switch (activation) {
        case ActivationType::Relu:
            return relu_derivative(x);
        default:
            return relu_derivative(x);
    }
}

std::vector<double> Layer::forward(const std::vector<double>& input) {
    last_input = input;
    last_z.assign(output_size, 0.0);
    std::vector<double> output(output_size, 0.0);
    
    for (int i = 0; i < output_size; ++i) {
        double sum = biases[i];
        for (int j = 0; j < input_size; ++j) {
            sum += weights[i][j] * input[j];
        }
        last_z[i] = sum;
        output[i] = activate(sum);
    }
    
    return output;
}

std::vector<double> Layer::backward(const std::vector<double>& output_gradient) {
    std::vector<double> input_gradient(input_size, 0.0);
    
    for (int i = 0; i < output_size; ++i) {
        double delta = output_gradient[i] * activation_derivative(last_z[i]);
        
        // Calculate input gradient for the previous layer
        for (int j = 0; j < input_size; ++j) {
            input_gradient[j] += weights[i][j] * delta;
        }
        
        // Accumulate gradients
        for (int j = 0; j < input_size; ++j) {
            weight_grads[i][j] += delta * last_input[j];
        }
        bias_grads[i] += delta;
    }
    
    return input_gradient;
}

void Layer::update(double learning_rate) {
    for (int i = 0; i < output_size; ++i) {
        for (int j = 0; j < input_size; ++j) {
            weights[i][j] -= learning_rate * weight_grads[i][j];
        }
        biases[i] -= learning_rate * bias_grads[i];
    }
}

void Layer::zero_grad() {
    for (int i = 0; i < output_size; ++i) {
        std::fill(weight_grads[i].begin(), weight_grads[i].end(), 0.0);
    }
    std::fill(bias_grads.begin(), bias_grads.end(), 0.0);
}

std::vector<std::vector<double>>& Layer::get_weights() {
    return weights;
}

std::vector<double>& Layer::get_biases() {
    return biases;
}

int Layer::get_input_size() const {
    return input_size;
}

int Layer::get_output_size() const {
    return output_size;
}
