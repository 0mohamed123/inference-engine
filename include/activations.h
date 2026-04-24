#pragma once
#include <vector>
#include <cmath>
#include <algorithm>

namespace activations {

    std::vector<double> relu(const std::vector<double>& x) {
        std::vector<double> out(x.size());
        for (size_t i = 0; i < x.size(); i++)
            out[i] = std::max(0.0, x[i]);
        return out;
    }

    std::vector<double> sigmoid(const std::vector<double>& x) {
        std::vector<double> out(x.size());
        for (size_t i = 0; i < x.size(); i++)
            out[i] = 1.0 / (1.0 + std::exp(-x[i]));
        return out;
    }

    std::vector<double> softmax(const std::vector<double>& x) {
        std::vector<double> out(x.size());
        double max_val = *std::max_element(x.begin(), x.end());
        double sum = 0.0;
        for (size_t i = 0; i < x.size(); i++) {
            out[i] = std::exp(x[i] - max_val);
            sum += out[i];
        }
        for (auto& v : out) v /= sum;
        return out;
    }

    std::vector<double> tanh_act(const std::vector<double>& x) {
        std::vector<double> out(x.size());
        for (size_t i = 0; i < x.size(); i++)
            out[i] = std::tanh(x[i]);
        return out;
    }

    int argmax(const std::vector<double>& x) {
        return std::max_element(x.begin(), x.end()) - x.begin();
    }
}