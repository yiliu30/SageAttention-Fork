#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

#include <cutlass/half.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kThroughputThreads = 256;
constexpr int kWarpsPerThroughputBlock = kThroughputThreads / 32;
constexpr int kFoldChains = 4;
constexpr double kOpsPerMma = 16.0 * 8.0 * 32.0 * 2.0;
constexpr uint32_t kE4m3OneOver64 = 0x08080808u;

#define CUDA_CHECK(call)                                                        \
  do {                                                                          \
    cudaError_t status_ = (call);                                                \
    if (status_ != cudaSuccess) {                                                \
      throw std::runtime_error(std::string(#call) + ": " +                      \
                               cudaGetErrorString(status_));                     \
    }                                                                           \
  } while (0)

struct Options {
  int warmup = 5;
  int iterations = 21;
  int latency_inner = 65536;
  int inner = 8192;
  int folds = 4096;
  int blocks_per_sm = 60;
  int chains = 4;
  int fold_k = 64;
  std::string mode = "all";
  bool profile = false;
};

struct Result {
  std::string label;
  int registers = 0;
  int active_blocks_per_sm = 0;
  int launched_blocks = 0;
  int threads = 0;
  double median_ms = 0.0;
  double mma_instructions = 0.0;
  double mma_per_second = 0.0;
  double tensor_tops = 0.0;
  double checksum = 0.0;
  double cycles_per_instruction = std::numeric_limits<double>::quiet_NaN();
  double ns_per_instruction = std::numeric_limits<double>::quiet_NaN();
};

__device__ __forceinline__ void mma_e4m3_f32(
    float& d0, float& d1, float& d2, float& d3, uint32_t a0, uint32_t a1,
    uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
      "{%0, %1, %2, %3}, "
      "{%4, %5, %6, %7}, "
      "{%8, %9}, "
      "{%0, %1, %2, %3};\n"
      : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void mma_e4m3_f16(
    uint32_t& d0, uint32_t& d1, uint32_t a0, uint32_t a1, uint32_t a2,
    uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f16.e4m3.e4m3.f16 "
      "{%0, %1}, "
      "{%2, %3, %4, %5}, "
      "{%6, %7}, "
      "{%0, %1};\n"
      : "+r"(d0), "+r"(d1)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ float half_bits_to_float(uint16_t bits) {
  return static_cast<float>(cutlass::half_t::bitcast(bits));
}

__device__ __forceinline__ float sum_f16_accumulator(uint32_t d0,
                                                      uint32_t d1) {
  return half_bits_to_float(static_cast<uint16_t>(d0)) +
         half_bits_to_float(static_cast<uint16_t>(d0 >> 16)) +
         half_bits_to_float(static_cast<uint16_t>(d1)) +
         half_bits_to_float(static_cast<uint16_t>(d1 >> 16));
}

template <bool UseFp16>
__global__ __launch_bounds__(32)
void dependent_latency_kernel(int inner, float* output,
                              unsigned long long* cycles) {
  const uint32_t a0 = kE4m3OneOver64;
  const uint32_t a1 = kE4m3OneOver64;
  const uint32_t a2 = kE4m3OneOver64;
  const uint32_t a3 = kE4m3OneOver64;
  const uint32_t b0 = kE4m3OneOver64;
  const uint32_t b1 = kE4m3OneOver64;

  __syncwarp();
  const unsigned long long start = clock64();

  float checksum = 0.0f;
  if constexpr (UseFp16) {
    uint32_t d0 = 0;
    uint32_t d1 = 0;
    for (int repeat = 0; repeat < inner; ++repeat) {
      mma_e4m3_f16(d0, d1, a0, a1, a2, a3, b0, b1);
    }
    checksum = sum_f16_accumulator(d0, d1);
  } else {
    float d0 = 0.0f;
    float d1 = 0.0f;
    float d2 = 0.0f;
    float d3 = 0.0f;
    for (int repeat = 0; repeat < inner; ++repeat) {
      mma_e4m3_f32(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1);
    }
    checksum = d0 + d1 + d2 + d3;
  }

  __syncwarp();
  const unsigned long long stop = clock64();
  if (threadIdx.x == 0) {
    output[blockIdx.x] = checksum;
    cycles[blockIdx.x] = stop - start;
  }
}

template <bool UseFp16, int Chains>
__global__ __launch_bounds__(kThroughputThreads)
void raw_throughput_kernel(int inner, float* output) {
  const uint32_t a0 = kE4m3OneOver64;
  const uint32_t a1 = kE4m3OneOver64;
  const uint32_t a2 = kE4m3OneOver64;
  const uint32_t a3 = kE4m3OneOver64;
  const uint32_t b0 = kE4m3OneOver64;
  const uint32_t b1 = kE4m3OneOver64;

  float checksum = 0.0f;
  if constexpr (UseFp16) {
    uint32_t accum[Chains][2] = {};
    for (int repeat = 0; repeat < inner; ++repeat) {
#pragma unroll
      for (int chain = 0; chain < Chains; ++chain) {
        mma_e4m3_f16(accum[chain][0], accum[chain][1], a0, a1, a2, a3, b0,
                     b1);
      }
    }
#pragma unroll
    for (int chain = 0; chain < Chains; ++chain) {
      checksum += sum_f16_accumulator(accum[chain][0], accum[chain][1]);
    }
  } else {
    float accum[Chains][4] = {};
    for (int repeat = 0; repeat < inner; ++repeat) {
#pragma unroll
      for (int chain = 0; chain < Chains; ++chain) {
        mma_e4m3_f32(accum[chain][0], accum[chain][1], accum[chain][2],
                     accum[chain][3], a0, a1, a2, a3, b0, b1);
      }
    }
#pragma unroll
    for (int chain = 0; chain < Chains; ++chain) {
      checksum += accum[chain][0] + accum[chain][1] + accum[chain][2] +
                  accum[chain][3];
    }
  }

  const size_t index =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  output[index] = checksum;
}

template <bool UseFp16, int FoldMmas>
__global__ __launch_bounds__(kThroughputThreads)
void folded_accumulator_kernel(int folds, const uint32_t* operands,
                               float* output) {
  const uint32_t a0 = kE4m3OneOver64;
  const uint32_t a1 = kE4m3OneOver64;
  const uint32_t a2 = kE4m3OneOver64;
  const uint32_t a3 = kE4m3OneOver64;
  uint32_t b0[kFoldChains];
  uint32_t b1[kFoldChains];
  // Runtime-distinct, rotating valid E4M3 words keep every intended partial
  // MMA live; identical constants can be coalesced across short fold chains.
#pragma unroll
  for (int chain = 0; chain < kFoldChains; ++chain) {
    b0[chain] = operands[2 * chain];
    b1[chain] = operands[2 * chain + 1];
  }

  float persistent[kFoldChains][4] = {};

  if constexpr (UseFp16) {
    for (int repeat = 0; repeat < folds; ++repeat) {
      uint32_t partial[kFoldChains][2] = {};
#pragma unroll
      for (int k_step = 0; k_step < FoldMmas; ++k_step) {
#pragma unroll
        for (int chain = 0; chain < kFoldChains; ++chain) {
          mma_e4m3_f16(partial[chain][0], partial[chain][1], a0, a1, a2, a3,
                       b0[chain], b1[chain]);
        }
      }
#pragma unroll
      for (int chain = 0; chain < kFoldChains; ++chain) {
        persistent[chain][0] +=
            half_bits_to_float(static_cast<uint16_t>(partial[chain][0]));
        persistent[chain][1] +=
            half_bits_to_float(static_cast<uint16_t>(partial[chain][0] >> 16));
        persistent[chain][2] +=
            half_bits_to_float(static_cast<uint16_t>(partial[chain][1]));
        persistent[chain][3] +=
            half_bits_to_float(static_cast<uint16_t>(partial[chain][1] >> 16));
        b0[chain] = (b0[chain] >> 8) | (b0[chain] << 24);
        b1[chain] = (b1[chain] >> 8) | (b1[chain] << 24);
      }
    }
  } else {
    for (int repeat = 0; repeat < folds; ++repeat) {
#pragma unroll
      for (int k_step = 0; k_step < FoldMmas; ++k_step) {
#pragma unroll
        for (int chain = 0; chain < kFoldChains; ++chain) {
          mma_e4m3_f32(persistent[chain][0], persistent[chain][1],
                       persistent[chain][2], persistent[chain][3], a0, a1, a2,
                       a3, b0[chain], b1[chain]);
        }
      }
#pragma unroll
      for (int chain = 0; chain < kFoldChains; ++chain) {
        b0[chain] = (b0[chain] >> 8) | (b0[chain] << 24);
        b1[chain] = (b1[chain] >> 8) | (b1[chain] << 24);
      }
    }
  }

  float checksum = 0.0f;
#pragma unroll
  for (int chain = 0; chain < kFoldChains; ++chain) {
    checksum += persistent[chain][0] + persistent[chain][1] +
                persistent[chain][2] + persistent[chain][3];
  }
  const size_t index =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  output[index] = checksum;
}

template <typename Launch, typename AfterSample>
double time_kernel(Launch&& launch, AfterSample&& after_sample, int warmup,
                   int iterations) {
  for (int i = 0; i < warmup; ++i) {
    launch();
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start;
  cudaEvent_t stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  std::vector<float> times;
  times.reserve(iterations);

  for (int i = 0; i < iterations; ++i) {
    CUDA_CHECK(cudaEventRecord(start));
    launch();
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventSynchronize(stop));
    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    times.push_back(elapsed_ms);
    after_sample();
  }

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  std::sort(times.begin(), times.end());
  return times[times.size() / 2];
}

double validate_output(float* device_output, size_t count) {
  std::vector<float> host_output(count);
  CUDA_CHECK(cudaMemcpy(host_output.data(), device_output,
                        count * sizeof(float), cudaMemcpyDeviceToHost));
  double checksum = 0.0;
  for (float value : host_output) {
    if (!std::isfinite(value)) {
      throw std::runtime_error("non-finite benchmark output");
    }
    checksum += static_cast<double>(value);
  }
  if (!std::isfinite(checksum) || checksum == 0.0) {
    throw std::runtime_error("invalid zero or non-finite checksum");
  }
  return checksum;
}

template <typename Kernel>
void get_kernel_metadata(Kernel kernel, int threads, int* registers,
                         int* active_blocks_per_sm) {
  cudaFuncAttributes attributes{};
  CUDA_CHECK(cudaFuncGetAttributes(&attributes,
                                   reinterpret_cast<const void*>(kernel)));
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      active_blocks_per_sm, kernel, threads, 0));
  *registers = attributes.numRegs;
}

void finish_rates(Result* result) {
  const double seconds = result->median_ms * 1.0e-3;
  result->mma_per_second = result->mma_instructions / seconds;
  result->tensor_tops = result->mma_per_second * kOpsPerMma / 1.0e12;
}

template <bool UseFp16>
Result run_latency(const Options& options, int sm_count) {
  constexpr int threads = 32;
  const int blocks = sm_count;
  float* output = nullptr;
  unsigned long long* cycles = nullptr;
  CUDA_CHECK(cudaMalloc(&output, blocks * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&cycles, blocks * sizeof(unsigned long long)));

  auto launch = [&] {
    dependent_latency_kernel<UseFp16><<<blocks, threads>>>(
        options.latency_inner, output, cycles);
  };

  Result result;
  result.label = UseFp16 ? "dependent-f16" : "dependent-f32";
  result.launched_blocks = blocks;
  result.threads = threads;
  get_kernel_metadata(dependent_latency_kernel<UseFp16>, threads,
                      &result.registers, &result.active_blocks_per_sm);
  std::vector<unsigned long long> host_cycles(blocks);
  std::vector<double> cycle_samples;
  cycle_samples.reserve(options.iterations);
  auto collect_cycles = [&] {
    CUDA_CHECK(cudaMemcpy(host_cycles.data(), cycles,
                          blocks * sizeof(unsigned long long),
                          cudaMemcpyDeviceToHost));
    std::sort(host_cycles.begin(), host_cycles.end());
    cycle_samples.push_back(
        static_cast<double>(host_cycles[host_cycles.size() / 2]) /
        options.latency_inner);
  };
  result.median_ms = time_kernel(launch, collect_cycles, options.warmup,
                                 options.iterations);
  result.mma_instructions =
      static_cast<double>(blocks) * options.latency_inner;
  finish_rates(&result);
  result.ns_per_instruction =
      result.median_ms * 1.0e6 / options.latency_inner;

  std::sort(cycle_samples.begin(), cycle_samples.end());
  result.cycles_per_instruction =
      cycle_samples[cycle_samples.size() / 2];
  result.checksum = validate_output(output, blocks);

  CUDA_CHECK(cudaFree(cycles));
  CUDA_CHECK(cudaFree(output));
  return result;
}

template <bool UseFp16, int Chains>
Result run_raw(const Options& options, int sm_count) {
  const int blocks = sm_count * options.blocks_per_sm;
  const size_t output_count =
      static_cast<size_t>(blocks) * kThroughputThreads;
  float* output = nullptr;
  CUDA_CHECK(cudaMalloc(&output, output_count * sizeof(float)));

  auto launch = [&] {
    raw_throughput_kernel<UseFp16, Chains>
        <<<blocks, kThroughputThreads>>>(options.inner, output);
  };

  Result result;
  result.label = std::string("raw-f") + (UseFp16 ? "16-c" : "32-c") +
                 std::to_string(Chains);
  result.launched_blocks = blocks;
  result.threads = kThroughputThreads;
  get_kernel_metadata(raw_throughput_kernel<UseFp16, Chains>,
                      kThroughputThreads, &result.registers,
                      &result.active_blocks_per_sm);

  if (options.profile) {
    for (int i = 0; i < options.warmup; ++i) {
      launch();
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaProfilerStart());
    launch();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaProfilerStop());
    result.checksum = validate_output(output, output_count);
    CUDA_CHECK(cudaFree(output));
    return result;
  }

  result.median_ms =
      time_kernel(launch, [] {}, options.warmup, options.iterations);
  result.mma_instructions =
      static_cast<double>(blocks) * kWarpsPerThroughputBlock * options.inner *
      Chains;
  finish_rates(&result);
  result.checksum = validate_output(output, output_count);
  CUDA_CHECK(cudaFree(output));
  return result;
}

template <bool UseFp16, int FoldMmas>
Result run_folded(const Options& options, int sm_count) {
  constexpr std::array<uint32_t, 2 * kFoldChains> host_operands = {
      0x08090a0bu, 0x0c0d0e0fu, 0x10111213u, 0x14151617u,
      0x18191a1bu, 0x1c1d1e1fu, 0x20212223u, 0x24252627u};
  const int blocks = sm_count * options.blocks_per_sm;
  const size_t output_count =
      static_cast<size_t>(blocks) * kThroughputThreads;
  float* output = nullptr;
  uint32_t* operands = nullptr;
  CUDA_CHECK(cudaMalloc(&output, output_count * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&operands, host_operands.size() * sizeof(uint32_t)));
  CUDA_CHECK(cudaMemcpy(operands, host_operands.data(),
                        host_operands.size() * sizeof(uint32_t),
                        cudaMemcpyHostToDevice));

  auto launch = [&] {
    folded_accumulator_kernel<UseFp16, FoldMmas>
        <<<blocks, kThroughputThreads>>>(options.folds, operands, output);
  };

  Result result;
  constexpr int fold_k = FoldMmas * 32;
  result.label = std::string("folded-f") + (UseFp16 ? "16-k" : "32-k") +
                 std::to_string(fold_k);
  result.launched_blocks = blocks;
  result.threads = kThroughputThreads;
  get_kernel_metadata(folded_accumulator_kernel<UseFp16, FoldMmas>,
                      kThroughputThreads, &result.registers,
                      &result.active_blocks_per_sm);
  result.median_ms =
      time_kernel(launch, [] {}, options.warmup, options.iterations);
  result.mma_instructions =
      static_cast<double>(blocks) * kWarpsPerThroughputBlock * options.folds *
      FoldMmas * kFoldChains;
  finish_rates(&result);
  result.checksum = validate_output(output, output_count);
  CUDA_CHECK(cudaFree(operands));
  CUDA_CHECK(cudaFree(output));
  return result;
}

void print_result_header() {
  std::cout << std::left << std::setw(18) << "mode" << std::right
            << std::setw(7) << "regs" << std::setw(9) << "max_blk"
            << std::setw(9) << "max_wrp" << std::setw(9) << "grid"
            << std::setw(12) << "median_ms" << std::setw(15) << "MMA_Ginst/s"
            << std::setw(14) << "tensor_TOPS" << std::setw(17) << "checksum"
            << '\n';
}

void print_result(const Result& result) {
  const int active_warps =
      result.active_blocks_per_sm * result.threads / 32;
  std::cout << std::left << std::setw(18) << result.label << std::right
            << std::setw(7) << result.registers << std::setw(9)
            << result.active_blocks_per_sm << std::setw(9) << active_warps
            << std::setw(9) << result.launched_blocks << std::fixed
            << std::setprecision(4) << std::setw(12) << result.median_ms
            << std::setprecision(3) << std::setw(15)
            << result.mma_per_second / 1.0e9 << std::setw(14)
            << result.tensor_tops << std::scientific << std::setprecision(6)
            << std::setw(17) << result.checksum << std::defaultfloat << '\n';
}

int parse_positive_int(const char* option, const char* value) {
  char* end = nullptr;
  const long parsed = std::strtol(value, &end, 10);
  if (end == value || *end != '\0' || parsed <= 0 ||
      parsed > std::numeric_limits<int>::max()) {
    throw std::runtime_error(std::string("invalid value for ") + option +
                             ": " + value);
  }
  return static_cast<int>(parsed);
}

void print_usage(const char* program) {
  std::cout
      << "Usage: " << program << " [options]\n"
      << "  --mode all|latency|raw|fold|raw-f16|raw-f32|fold-f16|fold-f32\n"
      << "  --chains 1|2|4|8       single raw-mode chain count (default 4)\n"
      << "  --fold-k 64|128         single fold-mode K interval (default 64)\n"
      << "  --warmup N              warmup launches (default 5)\n"
      << "  --iterations N          timed launches, median reported (default 21)\n"
      << "  --latency-inner N       dependent MMAs per warp (default 65536)\n"
      << "  --inner N               raw loop repeats (default 8192)\n"
      << "  --folds N               realistic fold repeats (default 4096)\n"
      << "  --blocks-per-sm N       throughput grid multiplier (default 60)\n"
      << "  --profile               one cudaProfilerStart/Stop launch; raw-f16 or "
         "raw-f32 only\n";
}

Options parse_options(int argc, char** argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    auto require_value = [&]() -> const char* {
      if (++index >= argc) {
        throw std::runtime_error("missing value after " + argument);
      }
      return argv[index];
    };
    if (argument == "--mode") {
      options.mode = require_value();
    } else if (argument == "--chains") {
      options.chains = parse_positive_int("--chains", require_value());
    } else if (argument == "--fold-k") {
      options.fold_k = parse_positive_int("--fold-k", require_value());
    } else if (argument == "--warmup") {
      options.warmup = parse_positive_int("--warmup", require_value());
    } else if (argument == "--iterations") {
      options.iterations =
          parse_positive_int("--iterations", require_value());
    } else if (argument == "--latency-inner") {
      options.latency_inner =
          parse_positive_int("--latency-inner", require_value());
    } else if (argument == "--inner") {
      options.inner = parse_positive_int("--inner", require_value());
    } else if (argument == "--folds") {
      options.folds = parse_positive_int("--folds", require_value());
    } else if (argument == "--blocks-per-sm") {
      options.blocks_per_sm =
          parse_positive_int("--blocks-per-sm", require_value());
    } else if (argument == "--profile") {
      options.profile = true;
    } else if (argument == "--help" || argument == "-h") {
      print_usage(argv[0]);
      std::exit(0);
    } else {
      throw std::runtime_error("unknown option: " + argument);
    }
  }

  const std::array<std::string, 8> valid_modes = {
      "all",      "latency", "raw",      "fold",
      "raw-f16",  "raw-f32", "fold-f16", "fold-f32"};
  if (std::find(valid_modes.begin(), valid_modes.end(), options.mode) ==
      valid_modes.end()) {
    throw std::runtime_error("invalid --mode: " + options.mode);
  }
  if (options.chains != 1 && options.chains != 2 && options.chains != 4 &&
      options.chains != 8) {
    throw std::runtime_error("--chains must be 1, 2, 4, or 8");
  }
  if (options.fold_k != 64 && options.fold_k != 128) {
    throw std::runtime_error("--fold-k must be 64 or 128");
  }
  if (options.profile && options.mode != "raw-f16" &&
      options.mode != "raw-f32") {
    throw std::runtime_error(
        "--profile requires --mode raw-f16 or --mode raw-f32");
  }
  return options;
}

template <bool UseFp16>
Result dispatch_raw(const Options& options, int sm_count) {
  switch (options.chains) {
    case 1:
      return run_raw<UseFp16, 1>(options, sm_count);
    case 2:
      return run_raw<UseFp16, 2>(options, sm_count);
    case 4:
      return run_raw<UseFp16, 4>(options, sm_count);
    case 8:
      return run_raw<UseFp16, 8>(options, sm_count);
    default:
      throw std::runtime_error("unreachable chain count");
  }
}

template <bool UseFp16>
Result dispatch_folded(const Options& options, int sm_count) {
  if (options.fold_k == 64) {
    return run_folded<UseFp16, 2>(options, sm_count);
  }
  return run_folded<UseFp16, 4>(options, sm_count);
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
    if (properties.major != 12 || properties.minor != 0) {
      throw std::runtime_error("benchmark requires an SM120 GPU");
    }
    if (options.blocks_per_sm >
        std::numeric_limits<int>::max() / properties.multiProcessorCount) {
      throw std::runtime_error("--blocks-per-sm makes the grid too large");
    }
    const int throughput_blocks =
        options.blocks_per_sm * properties.multiProcessorCount;
    if (throughput_blocks > properties.maxGridSize[0]) {
      throw std::runtime_error("--blocks-per-sm exceeds the CUDA grid limit");
    }

    std::cout << "GPU: " << properties.name << ", SM "
              << properties.major << properties.minor << ", "
              << properties.multiProcessorCount << " SMs, "
              << properties.maxThreadsPerMultiProcessor
              << " max threads/SM\n"
              << "Timing: CUDA-event median, warmup=" << options.warmup
              << ", iterations=" << options.iterations
              << "; output stores are timed but excluded from MMA work\n"
              << "Operands: E4M3 2^-6; one native m16n8k32 MMA = "
              << static_cast<long long>(kOpsPerMma) << " tensor operations\n";

    if (options.profile) {
      const Result result = options.mode == "raw-f16"
                                ? dispatch_raw<true>(
                                      options, properties.multiProcessorCount)
                                : dispatch_raw<false>(
                                      options, properties.multiProcessorCount);
      std::cout << "Profile launch complete: " << result.label
                << ", registers/thread=" << result.registers
                << ", active blocks/SM=" << result.active_blocks_per_sm
                << ", grid=" << result.launched_blocks
                << ", checksum=" << std::scientific << result.checksum
                << '\n';
      return 0;
    }

    if (options.mode == "all" || options.mode == "latency") {
      std::cout << "\nDEPENDENT CHAIN (one warp block per SM)\n";
      print_result_header();
      const Result f32 =
          run_latency<false>(options, properties.multiProcessorCount);
      const Result f16 =
          run_latency<true>(options, properties.multiProcessorCount);
      print_result(f32);
      print_result(f16);
      std::cout << std::fixed << std::setprecision(3)
                << "dependent-f32: " << f32.cycles_per_instruction
                << " cycles/instruction, " << f32.ns_per_instruction
                << " ns/instruction\n"
                << "dependent-f16: " << f16.cycles_per_instruction
                << " cycles/instruction, " << f16.ns_per_instruction
                << " ns/instruction\n";
    }

    if (options.mode == "all" || options.mode == "raw") {
      std::cout << "\nRAW SUSTAINED THROUGHPUT\n";
      print_result_header();
      std::array<Result, 4> f32_results;
      std::array<Result, 4> f16_results;
      const std::array<int, 4> chains = {1, 2, 4, 8};
      for (size_t index = 0; index < chains.size(); ++index) {
        Options selected = options;
        selected.chains = chains[index];
        f32_results[index] =
            dispatch_raw<false>(selected, properties.multiProcessorCount);
        f16_results[index] =
            dispatch_raw<true>(selected, properties.multiProcessorCount);
        print_result(f32_results[index]);
        print_result(f16_results[index]);
      }
      std::cout << "FP16/FP32 throughput speedup:";
      for (size_t index = 0; index < chains.size(); ++index) {
        std::cout << " c" << chains[index] << '=' << std::fixed
                  << std::setprecision(3)
                  << f16_results[index].tensor_tops /
                         f32_results[index].tensor_tops
                  << 'x';
      }
      std::cout << '\n';
    } else if (options.mode == "raw-f16" ||
               options.mode == "raw-f32") {
      std::cout << "\nRAW SUSTAINED THROUGHPUT\n";
      print_result_header();
      const Result result =
          options.mode == "raw-f16"
              ? dispatch_raw<true>(options, properties.multiProcessorCount)
              : dispatch_raw<false>(options, properties.multiProcessorCount);
      print_result(result);
    }

    if (options.mode == "all" || options.mode == "fold") {
      std::cout << "\nFOLDED FP16 PARTIALS VS FP32 PERSISTENT (four N slices)\n";
      print_result_header();
      std::array<Result, 2> f32_results;
      std::array<Result, 2> f16_results;
      const std::array<int, 2> fold_k = {64, 128};
      for (size_t index = 0; index < fold_k.size(); ++index) {
        Options selected = options;
        selected.fold_k = fold_k[index];
        f32_results[index] =
            dispatch_folded<false>(selected, properties.multiProcessorCount);
        f16_results[index] =
            dispatch_folded<true>(selected, properties.multiProcessorCount);
        print_result(f32_results[index]);
        print_result(f16_results[index]);
      }
      std::cout << "Folded FP16/FP32 throughput speedup:";
      for (size_t index = 0; index < fold_k.size(); ++index) {
        std::cout << " k" << fold_k[index] << '=' << std::fixed
                  << std::setprecision(3)
                  << f16_results[index].tensor_tops /
                         f32_results[index].tensor_tops
                  << 'x';
      }
      std::cout << '\n';
    } else if (options.mode == "fold-f16" ||
               options.mode == "fold-f32") {
      std::cout << "\nFOLDED FP16 PARTIALS VS FP32 PERSISTENT\n";
      print_result_header();
      const Result result =
          options.mode == "fold-f16"
              ? dispatch_folded<true>(options, properties.multiProcessorCount)
              : dispatch_folded<false>(options,
                                       properties.multiProcessorCount);
      print_result(result);
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
