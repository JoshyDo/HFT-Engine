#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <string_view>
#include <random>
#include "rdtsc_timer.hpp"
#include "parsers.hpp" 
#include "market_tick.hpp"

// 1. Der Compiler-Defeater
template <class T>
__attribute__((always_inline)) inline void DO_NOT_OPTIMIZE(T& value) {
    __asm__ volatile("" : "+r,m"(value) : : "memory");
}

// 2. Die Virtual-Class Konkurrenz (Das Anti-Pattern)
class IVirtualParser {
public:
    virtual ~IVirtualParser() = default;
    virtual int parse(std::string_view raw, MarketTick& out, std::string_view& symbol_out) = 0;
};

class VirtualSBEParser : public IVirtualParser {
public:
    int parse(std::string_view raw, MarketTick& out, std::string_view& symbol_out) override {
        if (raw.size() < sizeof(phase7::SbeMessageHeader) + sizeof(phase7::SbeTickPayload)) {
            return 1; 
        }

        const auto* header = reinterpret_cast<const phase7::SbeMessageHeader*>(raw.data());
        if (header->templateId != 42) { 
            return 1;
        }

        const auto* payload = reinterpret_cast<const phase7::SbeTickPayload*>(raw.data() + sizeof(phase7::SbeMessageHeader));

        size_t sym_len = 0;
        while (sym_len < 8 && payload->symbol[sym_len] != '\0') {
            ++sym_len;
        }
        symbol_out = std::string_view(payload->symbol, sym_len);

        out.bid = payload->bid;
        out.ask = payload->ask;
        
        const double spread = out.ask - out.bid;
        out.weight = (spread > 0.0) ? (1.0 / spread) : 1.0;

        return 0; 
    }
};

// NEU: Der Poison-Parser (Simuliert z.B. Heartbeats oder Quotes)
class VirtualPoisonParser : public IVirtualParser {
public:
    int parse(std::string_view raw, MarketTick& out, [[maybe_unused]] std::string_view& symbol_out) override {
        // Minimal andere Logik, zwingt die CPU in einen anderen Instruktionspfad
        if (raw.size() < sizeof(uint32_t)) return 1;
        out.bid = *reinterpret_cast<const uint32_t*>(raw.data()) ^ 0xDEADBEEF;
        return 0;
    }
};


int main() {
    constexpr int ITERATIONS = 10'000'000;
    std::vector<uint64_t> crtp_cycles(ITERATIONS);
    std::vector<uint64_t> virtual_cycles(ITERATIONS);

    // Mock Payload preparieren (L1 Cache Warmup)
    alignas(64) char buffer[sizeof(phase7::SbeMessageHeader) + sizeof(phase7::SbeTickPayload)] = {0};
    
    auto* header = reinterpret_cast<phase7::SbeMessageHeader*>(buffer);
    header->blockLength = sizeof(phase7::SbeTickPayload);
    header->templateId = 42;
    header->schemaId = 1;
    header->version = 1;

    auto* payload = reinterpret_cast<phase7::SbeTickPayload*>(buffer + sizeof(phase7::SbeMessageHeader));
    const char* sym = "BTCUSDT";
    for (int i = 0; i < 7; ++i) payload->symbol[i] = sym[i];
    payload->symbol[7] = '\0';
    payload->bid = 42000.0;
    payload->ask = 42001.0;

    std::string_view raw_view(buffer, sizeof(buffer));

    // --- Branch Predictor Poisoning Setup ---
    std::vector<IVirtualParser*> mixed_parsers(ITERATIONS);
    VirtualSBEParser sbe_inst;
    VirtualPoisonParser poison_inst;
    
    std::mt19937 rng(42); // Fixer Seed für reproduzierbare Benchmarks
    std::uniform_int_distribution<int> dist(0, 1);
    
    for (int i = 0; i < ITERATIONS; ++i) {
        // Zerstört die Vorhersagbarkeit für die CPU komplett
        mixed_parsers[i] = dist(rng) ? static_cast<IVirtualParser*>(&sbe_inst) 
                                     : static_cast<IVirtualParser*>(&poison_inst);
    }

    RdtscTimer timer;
    phase7::SbeParser crtp_parser;
    
    MarketTick out_tick{};
    std::string_view out_sym;

    std::cout << "[RDTSC] Starte Micro-Benchmark (Poisoned Branch Predictor)...\n";

    // --- TEST A: CRTP (Statisch, ge-inlined, kein VTable) ---
    for (int i = 0; i < ITERATIONS; ++i) {
        timer.start();
        int res = crtp_parser.parse(raw_view, out_tick, out_sym); 
        timer.stop();
        
        DO_NOT_OPTIMIZE(res); 
        DO_NOT_OPTIMIZE(out_tick.bid);
        crtp_cycles[i] = timer.elapsed_cycles();
    }

    // --- TEST B: Virtual (VTable Lookup + Pipeline Flushes) ---
    for (int i = 0; i < ITERATIONS; ++i) {
        timer.start();
        int res = mixed_parsers[i]->parse(raw_view, out_tick, out_sym);
        timer.stop();
        
        DO_NOT_OPTIMIZE(res);
        DO_NOT_OPTIMIZE(out_tick.bid);
        virtual_cycles[i] = timer.elapsed_cycles();
    }

    // Auswertung
    auto print_stats = [](const std::string& name, std::vector<uint64_t>& cycles) {
        std::sort(cycles.begin(), cycles.end());
        double mean = std::accumulate(cycles.begin(), cycles.end(), 0.0) / cycles.size();
        uint64_t p999 = cycles[cycles.size() * 0.999];
        uint64_t min_cycles = cycles[0];
        
        std::cout << name << " -> Min: " << min_cycles << " | Mean: " << mean 
                  << " | 99.9th: " << p999 << " Cycles\n";
    };

    print_stats("CRTP Parser   ", crtp_cycles);
    print_stats("Virtual Parser", virtual_cycles);

    return 0;
}
