#pragma once
#include <vector>
#include <random>
#include <stdexcept>
#include "activations.h"

class DenseLayer {
public:
    int in_size, out_size;
    std::vector<std::vector<double>> weights;
    std::vector<double> bias;
    std::string activation;

    DenseLayer(int in_size, int out_size, const std::string& activation = "relu")
        : in_size(in_size), out_size(out_size), activation(activation) {
        std::mt19937 rng(42);
        std::normal_distribution<double> dist(0.0, std::sqrt(2.0 / in_size));
        weights.resize(out_size, std::vector<double>(in_size));
        bias.resize(out_size, 0.0);
        for (auto& row : weights)
            for (auto& w : row)
                w = dist(rng);
    }

    std::vector<double> forward(const std::vector<double>& input) {
        if ((int)input.size() != in_size)
            throw std::invalid_argument("Input size mismatch");

        std::vector<double> output(out_size, 0.0);
        for (int i = 0; i < out_size; i++) {
            for (int j = 0; j < in_size; j++)
                output[i] += weights[i][j] * input[j];
            output[i] += bias[i];
        }

        if (activation == "relu")    return activations::relu(output);
        if (activation == "sigmoid") return activations::sigmoid(output);
        if (activation == "softmax") return activations::softmax(output);
        if (activation == "tanh")    return activations::tanh_act(output);
        return output;
    }
};