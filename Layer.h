#pragma once

#include <vector>

enum class ActivationType {
    Relu
};

class Layer {
public:
    Layer(int input_size, int output_size, ActivationType activation = ActivationType::Relu);

    // Forward pass
    std::vector<double> forward(const std::vector<double>& input);

    // Backward pass (compute gradients only)
    std::vector<double> backward(const std::vector<double>& output_gradient);

    // Apply gradients and update parameters
    void update(double learning_rate);

    // Reset accumulated gradients
    void zero_grad();

    // Getters for weights and biases
    std::vector<std::vector<double>>& get_weights();
    std::vector<double>& get_biases();

    int get_input_size() const;
    int get_output_size() const;

private:
    int input_size;
    int output_size;
    std::vector<std::vector<double>> weights;
    std::vector<double> biases;

    // Cache for backpropagation
    std::vector<double> last_input;
    std::vector<double> last_z;

    // Activation function
    double relu(double x);
    double relu_derivative(double x);

    double activate(double x);
    double activation_derivative(double x);

    ActivationType activation;

    // Gradient buffers
    std::vector<std::vector<double>> weight_grads;
    std::vector<double> bias_grads;
};
