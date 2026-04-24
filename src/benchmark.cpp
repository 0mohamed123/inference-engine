#include <iostream>
#include <vector>
#include "../include/neural_net.h"

int main() {
    std::cout << "============================================\n";
    std::cout << "   Inference Engine - Latency Benchmark\n";
    std::cout << "============================================\n\n";

    struct Config { std::string name; std::vector<int> layers; };
    std::vector<Config> configs = {
        {"Small  (128->64->10)",        {128, 64, 10}},
        {"Medium (256->128->64->10)",   {256, 128, 64, 10}},
        {"Large  (512->256->128->10)",  {512, 256, 128, 10}},
    };

    for (auto& cfg : configs) {
        NeuralNet net;
        int in = 784;
        for (size_t i = 0; i < cfg.layers.size(); i++) {
            std::string act = (i == cfg.layers.size()-1) ? "softmax" : "relu";
            net.add(in, cfg.layers[i], act);
            in = cfg.layers[i];
        }

        std::vector<double> input(784, 0.5);
        auto result = net.benchmark(input, 1000);

        std::cout << cfg.name << "\n";
        std::cout << "  Latency:    " << result.avg_latency_us << " us\n";
        std::cout << "  Throughput: " << (int)result.throughput_per_sec << " inf/sec\n\n";
    }

    std::cout << "============================================\n";
    return 0;
}