// Copyright 2026 Antfly, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <jaccl/jaccl.h>

#include <cstdio>
#include <cstring>
#include <exception>
#include <memory>
#include <limits>
#include <stdexcept>

namespace {
thread_local char last_error[1024] = {};

template <typename F> int checked(F&& operation) noexcept {
  try {
    operation();
    last_error[0] = '\0';
    return 0;
  } catch (const std::exception& error) {
    std::snprintf(last_error, sizeof(last_error), "%s", error.what());
  } catch (...) {
    std::snprintf(last_error, sizeof(last_error), "unknown JACCL error");
  }
  return -1;
}

struct Handle {
  std::shared_ptr<jaccl::Group> group;
};
} // namespace

extern "C" {

const char* antfly_jaccl_last_error() noexcept { return last_error; }

int antfly_jaccl_open(int rank, const char* coordinator,
                      const char* devices_file, void** output) noexcept {
  if (!output || !coordinator || !devices_file || rank < 0 || rank > 1)
    return checked([] { throw std::invalid_argument("invalid JACCL open arguments"); });
  *output = nullptr;
  return checked([&] {
    auto config = jaccl::Config()
                      .set_rank(rank)
                      .set_coordinator(coordinator)
                      .set_devices_from_file(devices_file);
    if (!config.is_valid()) throw std::runtime_error("invalid JACCL topology");
    auto group = jaccl::init(config, true);
    if (!group || group->rank() != rank || group->size() != 2)
      throw std::runtime_error("expected a two-rank JACCL group");
    *output = new Handle{std::move(group)};
  });
}

int antfly_jaccl_rank(void* handle) noexcept {
  if (!handle) return -1;
  return static_cast<Handle*>(handle)->group->rank();
}

int antfly_jaccl_size(void* handle) noexcept {
  if (!handle) return -1;
  return static_cast<Handle*>(handle)->group->size();
}

int antfly_jaccl_all_sum_f32(void* handle, const float* input,
                             float* output, size_t count) noexcept {
  if (!handle || (!input && count) || (!output && count) ||
      count > std::numeric_limits<size_t>::max() / sizeof(float))
    return checked([] { throw std::invalid_argument("invalid JACCL sum buffers"); });
  return checked([&] {
    static_cast<Handle*>(handle)->group->all_sum(input, output,
                                                 count * sizeof(float),
                                                 jaccl::Float32);
  });
}

int antfly_jaccl_all_gather(void* handle, const void* input, void* output,
                            size_t n_bytes) noexcept {
  if (!handle || (!input && n_bytes) || (!output && n_bytes)) return -1;
  return checked([&] {
    static_cast<Handle*>(handle)->group->all_gather(input, output, n_bytes);
  });
}

int antfly_jaccl_barrier(void* handle) noexcept {
  if (!handle) return -1;
  return checked([&] { static_cast<Handle*>(handle)->group->barrier(); });
}

void antfly_jaccl_close(void* handle) noexcept {
  delete static_cast<Handle*>(handle);
}

} // extern "C"
