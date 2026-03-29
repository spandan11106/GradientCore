#include "../../include/gradientcore/platform.hpp"

// Linux/POSIX specific headers
#include <cstdint>
#include <cstdlib>
#include <sys/mman.h>
#include <sys/random.h>
#include <unistd.h>

namespace gradientcore {
namespace platform {

void exit(int32_t code) { std::exit(code); }

uint32_t page_size() { return static_cast<uint32_t>(sysconf(_SC_PAGESIZE)); }

void *mem_reserve(uint64_t size) {
  // mmap with PROT_NONE reserves virtual address space without allocating
  // physical RAM
  void *ptr =
      mmap(nullptr, size, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (ptr == MAP_FAILED) {
    return nullptr;
  }
  return ptr;
}

bool mem_commit(void *ptr, uint64_t size) {
  // mprotect changes the permissions of the reserved memory to allow reading
  // and writing
  return mprotect(ptr, size, PROT_READ | PROT_WRITE) == 0;
}

bool mem_decommit(void *ptr, uint64_t size) {
  // madvise tells the Linux kernel it can take the physical RAM back,
  // but keeps the virtual memory addresses reserved for us.
  return madvise(ptr, size, MADV_DONTNEED) == 0;
}

bool mem_release(void *ptr, uint64_t size) {
  // munmap completely hands the memory (both virtual and physical) back to the
  // OS
  return munmap(ptr, size) == 0;
}

void get_entropy(void *data, uint64_t size) {
  // getrandom is the modern Linux standard for cryptographically secure random
  // bytes
  getrandom(data, size, 0);
}

} // namespace platform
} // namespace gradientcore
