// Simple standalone CUDA runtime sample that mirrors the logic of
// at::cuda::getCurrentDeviceProperties(). Build with:
//   nvcc -o get_device_props examples/get_device_props.cu
// Run with an optional device index argument to override the current device.

#include <cuda_runtime.h>

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void check(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        std::cerr << msg << ": " << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

int parse_device_index(int argc, char **argv) {
    if (argc <= 1) {
        return -1;  // use whichever device is already current
    }
    try {
        return std::stoi(argv[1]);
    } catch (const std::exception &ex) {
        std::cerr << "Failed to parse device index from '" << argv[1]
                  << "': " << ex.what() << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

void print_props(const cudaDeviceProp &props, int index) {
    std::cout << "Device " << index << ": " << props.name << "\n"
              << "  SM version       : " << props.major << "." << props.minor << "\n"
              << "  MultiProcessors  : " << props.multiProcessorCount << "\n"
              << "  Max threads/MP   : " << props.maxThreadsPerMultiProcessor << "\n"
              << "  Max threads/block: " << props.maxThreadsPerBlock << "\n"
              << "  Shared mem/block : " << props.sharedMemPerBlock / 1024 << " KB\n"
              << "  Global mem       : " << props.totalGlobalMem / (1024 * 1024) << " MB\n"
              << std::flush;
}

}  // namespace

int main(int argc, char **argv) {
    int requested = parse_device_index(argc, argv);

    int device_count = 0;
    check(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount failed");
    if (device_count == 0) {
        std::cerr << "No CUDA devices found" << std::endl;
        return EXIT_FAILURE;
    }

    int current = 0;
    check(cudaGetDevice(&current), "cudaGetDevice failed");

    if (requested >= 0) {
        if (requested >= device_count) {
            std::cerr << "Requested device " << requested << " but only "
                      << device_count << " devices are available" << std::endl;
            return EXIT_FAILURE;
        }
        check(cudaSetDevice(requested), "cudaSetDevice failed");
        current = requested;
    }

    cudaDeviceProp props{};
    check(cudaGetDeviceProperties(&props, current), "cudaGetDeviceProperties failed");
    print_props(props, current);

    return EXIT_SUCCESS;
}
