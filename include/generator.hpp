#pragma once
#include <cstdint>
#include <random>

#include "graph.hpp"

class DataGenerator {
public:
    static void populateGraph(GraphArena& arena, uint32_t num_nodes) {
        for (uint32_t i = 0; i < num_nodes; ++i) {
            arena.addNode(i);
        }

        std::random_device                     random_device;
        std::mt19937                           engine(random_device());
        std::uniform_real_distribution<double> distribution(0.5, 1.5);

        for (uint32_t i = 0; i < num_nodes; ++i) {
            for (uint32_t j = 0; j < num_nodes; ++j) {
                if (i != j) {
                    double price = distribution(engine);

                    double bid = price - 0.01;
                    double ask = price + 0.01;

                    arena.addEdge(i, j, bid, ask);
                }
            }
        }
    }
};