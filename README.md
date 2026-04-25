# Real-time Inference Engine (C++)

![Language](https://img.shields.io/badge/Language-C%2B%2B17-blue)
![Tests](https://img.shields.io/badge/Tests-25%20passing-green)
![Throughput](https://img.shields.io/badge/Throughput-18K%20inf%2Fsec-orange)

Neural network inference engine built from scratch in C++17.
Supports fully-connected layers with ReLU, Sigmoid, Softmax, and Tanh activations.

## Benchmark Results

    Small  (784->128->64->10)
      Latency:    54.57 us
      Throughput: 18,324 inf/sec

    Medium (784->256->128->64->10)
      Latency:    119.15 us
      Throughput: 8,392 inf/sec

    Large  (784->512->256->128->10)
      Latency:    272.71 us
      Throughput: 3,666 inf/sec

## Quick Start

    git clone https://github.com/0mohamed123/inference-engine.git
    cd inference-engine

    # Run demo
    g++ -std=c++17 -O2 -I include -o main src/main.cpp
    ./main

    # Run benchmark
    g++ -std=c++17 -O2 -I include -o benchmark src/benchmark.cpp
    ./benchmark

    # Run tests
    g++ -std=c++17 -O2 -I include -o test_runner tests/test_engine.cpp
    ./test_runner

## Usage

    NeuralNet net;
    net.add(784, 128, "relu")
       .add(128, 64, "relu")
       .add(64, 10, "softmax");

    std::vector<double> input(784, 0.5);
    int predicted_class = net.classify(input);

    auto result = net.benchmark(input, 1000);
    std::cout << result.throughput_per_sec << " inf/sec\n";

## Architecture

    Input -> DenseLayer -> Activation -> DenseLayer -> Activation -> Output

    Activations: ReLU, Sigmoid, Softmax, Tanh
    Weight init: He initialization (sqrt(2/fan_in))

## Test Results

    25 passed | 0 failed

    Tests cover: ReLU, sigmoid, softmax, tanh,
    dense layer shape, forward pass, error handling,
    network architecture, softmax sum, benchmark metrics

## Technologies

- C++17
- STL (vector, chrono, random)
- He weight initialization
- No external dependencies