#include "include/gradientcore/base/arena.hpp"
#include "include/gradientcore/base/base.hpp"
#include "include/gradientcore/base/prng.hpp"

#include <cstdio>

using namespace gradientcore;

int main() {
  Arena *perm_arena = Arena::create(MiB(1024), MiB(1), false);
  if (perm_arena != nullptr) {
    perm_arena->destroy();
  }

  std::printf("Arena created successfully! Memory mapped.\n");

  std::printf("Random float: %f\n", prng::randf());
  std::printf("Random normal: %f\n", prng::std_norm());

  return 0;
}
