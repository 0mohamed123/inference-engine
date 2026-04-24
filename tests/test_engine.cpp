#include <iostream>
#include <cassert>
#include <cmath>
#include <string>
#include <numeric>
#include "../include/neural_net.h"
#include "../include/activations.h"

int passed = 0, failed = 0;

void test(bool condition, const std::string& name) {
    if (condition) { std::cout << "  [PASS] " << name << "\n"; passed++; }
    else { std::cout << "  [FAIL] " << name << "\n"; failed++; }
}

void test_activations() {
    std::cout << "\n[Activations]\n";

    std::vector<double> x = {-2, -1, 0, 1, 2};

    auto r = activations::relu(x);
    test(r[0] == 0 && r[1] == 0 && r[2] == 0, "ReLU zeros negatives");
    test(r[3] == 1 && r[4] == 2, "ReLU keeps positives");

    auto s = activations::sigmoid(x);
    test(s[2] == 0.5, "sigmoid(0) = 0.5");
    test(s[3] > 0.5 && s[4] > 0.5, "sigmoid positives > 0.5");
    test(s[0] < 0.5 && s[1] < 0.5, "sigmoid negatives < 0.5");

    auto sm = activations::softmax(x);
    double sum = 0;
    for (auto v : sm) sum += v;
    test(std::abs(sum - 1.0) < 1e-9, "softmax sums to 1");
    test(activations::argmax(sm) == 4, "softmax argmax correct");

    auto t = activations::tanh_act(x);
    test(std::abs(t[2]) < 1e-9, "tanh(0) = 0");
    test(t[3] > 0 && t[4] > 0, "tanh positives > 0");
}

void test_dense_layer() {
    std::cout << "\n[DenseLayer]\n";

    DenseLayer layer(4, 3, "relu");
    test(layer.in_size == 4, "in_size correct");
    test(layer.out_size == 3, "out_size correct");
    test((int)layer.weights.size() == 3, "weights shape correct");
    test((int)layer.weights[0].size() == 4, "weights cols correct");

    std::vector<double> input = {1, 2, 3, 4};
    auto output = layer.forward(input);
    test((int)output.size() == 3, "output size correct");

    for (auto v : output)
        test(v >= 0, "ReLU output non-negative");

    bool threw = false;
    try { layer.forward({1, 2}); } catch (...) { threw = true; }
    test(threw, "wrong input size throws");
}

void test_neural_net() {
    std::cout << "\n[NeuralNet]\n";

    NeuralNet net;
    net.add(4, 8, "relu")
       .add(8, 4, "relu")
       .add(4, 2, "softmax");

    test((int)net.layers.size() == 3, "correct number of layers");

    std::vector<double> input = {0.1, 0.2, 0.3, 0.4};
    auto output = net.predict(input);
    test((int)output.size() == 2, "output size correct");

    double sum = 0;
    for (auto v : output) sum += v;
    test(std::abs(sum - 1.0) < 1e-9, "softmax output sums to 1");

    int cls = net.classify(input);
    test(cls == 0 || cls == 1, "classify returns valid class");
}

void test_benchmark() {
    std::cout << "\n[Benchmark]\n";

    NeuralNet net;
    net.add(10, 8, "relu").add(8, 4, "softmax");

    std::vector<double> input(10, 0.5);
    auto result = net.benchmark(input, 100);

    test(result.avg_latency_us > 0, "latency positive");
    test(result.throughput_per_sec > 0, "throughput positive");
    test(result.n_samples == 100, "sample count correct");
}

int main() {
    std::cout << "Running all tests...\n";
    test_activations();
    test_dense_layer();
    test_neural_net();
    test_benchmark();

    std::cout << "\n==============================\n";
    std::cout << "  " << passed << " passed | " << failed << " failed\n";
    std::cout << "==============================\n";
    return failed > 0 ? 1 : 0;
}