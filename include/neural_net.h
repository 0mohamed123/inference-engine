#pragma once
#include <vector>
#include <memory>
#include <chrono>
#include <iostream>
#include "layers.h"

class NeuralNet {
public:
    std::vector<DenseLayer> layers;

    NeuralNet& add(int in_size, int out_size, const std::string& activation = "relu") {
        layers.emplace_back(in_size, out_size, activation);
        return *this;
    }

    std::vector<double> predict(const std::vector<double>& input) {
        std::vector<double> out = input;
        for (auto& layer : layers)
            out = layer.forward(out);
        return out;
    }

    int classify(const std::vector<double>& input) {
        return activations::argmax(predict(input));
    }

    struct BenchmarkResult {
        double avg_latency_us;
        double throughput_per_sec;
        int n_samples;
    };

    BenchmarkResult benchmark(const std::vector<double>& sample_input, int n = 1000) {
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < n; i++)
            predict(sample_input);
        auto end = std::chrono::high_resolution_clock::now();

        double total_us = std::chrono::duration<double, std::micro>(end - start).count();
        return {total_us / n, n / (total_us / 1e6), n};
    }

    void summary() {
        std::cout << "Neural Network Architecture:\n";
        for (size_t i = 0; i < layers.size(); i++) {
            std::cout << "  Layer " << i+1 << ": "
                      << layers[i].in_size << " -> "
                      << layers[i].out_size
                      << " (" << layers[i].activation << ")\n";
        }
    }
};