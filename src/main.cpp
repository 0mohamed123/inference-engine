#include <iostream>
#include <vector>
#include "../include/neural_net.h"

int main() {
    std::cout << "=== Real-time Inference Engine ===\n\n";

    // Build network: 784 -> 128 -> 64 -> 10 (MNIST-like)
    NeuralNet net;
    net.add(784, 128, "relu")
       .add(128, 64, "relu")
       .add(64, 10, "softmax");

    net.summary();

    // Sample inference
    std::vector<double> input(784, 0.5);
    auto output = net.predict(input);

    std::cout << "\nSample output (softmax):\n";
    for (size_t i = 0; i < output.size(); i++)
        std::cout << "  class " << i << ": " << output[i] << "\n";

    std::cout << "\nPredicted class: " << net.classify(input) << "\n";

    // Benchmark
    auto result = net.benchmark(input, 1000);
    std::cout << "\nBenchmark (1000 inferences):\n";
    std::cout << "  Avg latency:  " << result.avg_latency_us << " us\n";
    std::cout << "  Throughput:   " << result.throughput_per_sec << " inferences/sec\n";

    return 0;
}