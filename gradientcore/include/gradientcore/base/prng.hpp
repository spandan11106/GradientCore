#pragma once

#include <cstdint>

namespace gradientcore {
// Permuted congruential generator
// Based on https://www.pcg-random.org

class PRNG {
private:
  uint64_t state;
  uint64_t increment;

public:
  PRNG();
  PRNG(uint64_t init_state, uint64_t init_seq);

  void seed(uint64_t init_state, uint64_t init_seq);

  uint32_t rand();

  float randf(); // Generates a random number between 0 and 1

  float std_norm(); // Generates a standard normal distribution
};

namespace prng {
void seed(uint64_t init_state, uint64_t init_seq);
uint32_t rand();
float randf();
float std_norm();
} // namespace prng

} // namespace gradientcore
