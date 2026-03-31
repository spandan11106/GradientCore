// src/platform/platform.cpp

// __linux__ is automatically defined by the compiler if you are on Linux
#if defined(__linux__)
#include "plat_linux.cpp" // Using your exact filename from the screenshot

// _WIN32 is automatically defined if you are compiling on Windows
#elif defined(_WIN32)
#include "plat_win32.cpp"

// Fallback error if someone tries to compile on a Mac or unsupported OS
#else
#error "Unsupported platform! Could not determine the OS."
#endif
