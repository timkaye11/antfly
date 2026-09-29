// Copyright 2026 Antfly, Inc.
// SPDX-License-Identifier: Apache-2.0

// Two-rank, trusted-network transport implementing the JACCL bridge ABI.
#include <arpa/inet.h>
#include <cerrno>
#include <fcntl.h>
#include <netdb.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {
thread_local char last_error[1024] = {};
constexpr uint64_t protocol_magic = 0x4146544350310001ULL;
constexpr size_t max_payload = 4 * 1024 * 1024;

template <typename F> int checked(F&& operation) noexcept {
  try {
    operation();
    last_error[0] = '\0';
    return 0;
  } catch (const std::exception& error) {
    std::snprintf(last_error, sizeof(last_error), "%s", error.what());
  } catch (...) {
    std::snprintf(last_error, sizeof(last_error), "unknown TCP transport error");
  }
  return -1;
}

struct Socket {
  int fd = -1;
  explicit Socket(int value = -1) : fd(value) {}
  ~Socket() { if (fd >= 0) ::close(fd); }
  Socket(const Socket&) = delete;
  Socket& operator=(const Socket&) = delete;
  Socket(Socket&& other) noexcept : fd(other.fd) { other.fd = -1; }
};

void configure_socket(int fd) {
  timeval timeout{300, 0};
  if (::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout)) != 0 ||
      ::setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout)) != 0)
    throw std::runtime_error("cannot configure TCP socket timeout");
#ifdef SO_NOSIGPIPE
  int enabled = 1;
  if (::setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &enabled, sizeof(enabled)) != 0)
    throw std::runtime_error("cannot disable TCP SIGPIPE");
#endif
}

void send_all(int fd, const void* data, size_t length) {
  auto* bytes = static_cast<const uint8_t*>(data);
  while (length) {
    ssize_t sent = ::send(fd, bytes, length, 0);
    if (sent < 0 && errno == EINTR) continue;
    if (sent <= 0) throw std::runtime_error("TCP send failed or peer disconnected");
    bytes += sent;
    length -= static_cast<size_t>(sent);
  }
}

void recv_all(int fd, void* data, size_t length) {
  auto* bytes = static_cast<uint8_t*>(data);
  while (length) {
    ssize_t received = ::recv(fd, bytes, length, 0);
    if (received < 0 && errno == EINTR) continue;
    if (received <= 0) throw std::runtime_error("TCP receive failed or peer disconnected");
    bytes += received;
    length -= static_cast<size_t>(received);
  }
}

struct AddressList {
  addrinfo* first = nullptr;
  ~AddressList() { if (first) ::freeaddrinfo(first); }
  AddressList() = default;
  AddressList(const AddressList&) = delete;
  AddressList& operator=(const AddressList&) = delete;
  AddressList(AddressList&& other) noexcept : first(other.first) { other.first = nullptr; }
};

AddressList resolve(const char* endpoint) {
  std::string address(endpoint);
  std::string host, port;
  if (!address.empty() && address.front() == '[') {
    auto close = address.find(']');
    if (close == std::string::npos || close + 1 >= address.size() || address[close + 1] != ':')
      throw std::invalid_argument("coordinator must be host:port or [IPv6]:port");
    host = address.substr(1, close - 1);
    port = address.substr(close + 2);
  } else {
    auto colon = address.rfind(':');
    if (colon == std::string::npos || address.find(':') != colon)
      throw std::invalid_argument("coordinator must be host:port or [IPv6]:port");
    host = address.substr(0, colon);
    port = address.substr(colon + 1);
  }
  if (host.empty() || port.empty() || port.find_first_not_of("0123456789") != std::string::npos)
    throw std::invalid_argument("invalid TCP coordinator address");
  auto port_number = std::stoul(port);
  if (port_number == 0 || port_number > 65535)
    throw std::invalid_argument("invalid TCP coordinator port");
  addrinfo hints{};
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;
  AddressList result;
  int status = ::getaddrinfo(host.c_str(), port.c_str(), &hints, &result.first);
  if (status != 0) throw std::runtime_error(std::string("TCP address resolution failed: ") + gai_strerror(status));
  return result;
}

Socket connect_peer(int rank, const char* coordinator) {
  auto addresses = resolve(coordinator);
  if (rank == 0) {
    Socket listener;
    for (auto* address = addresses.first; address; address = address->ai_next) {
      Socket candidate(::socket(address->ai_family, address->ai_socktype, address->ai_protocol));
      if (candidate.fd < 0) continue;
      int enabled = 1;
      ::setsockopt(candidate.fd, SOL_SOCKET, SO_REUSEADDR, &enabled, sizeof(enabled));
      if (::bind(candidate.fd, address->ai_addr, address->ai_addrlen) == 0 &&
          ::listen(candidate.fd, 1) == 0) {
        listener.fd = candidate.fd;
        candidate.fd = -1;
        break;
      }
    }
    if (listener.fd < 0) throw std::runtime_error("cannot bind TCP coordinator");
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    while (std::chrono::steady_clock::now() < deadline) {
      fd_set ready;
      FD_ZERO(&ready);
      FD_SET(listener.fd, &ready);
      timeval timeout{1, 0};
      int selected = ::select(listener.fd + 1, &ready, nullptr, nullptr, &timeout);
      if (selected < 0) throw std::runtime_error("TCP accept wait failed");
      if (selected > 0) {
        Socket peer(::accept(listener.fd, nullptr, nullptr));
        if (peer.fd < 0) throw std::runtime_error("TCP accept failed");
        configure_socket(peer.fd);
        return peer;
      }
    }
    throw std::runtime_error("timed out waiting for TCP rank 1");
  }
  auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
  while (std::chrono::steady_clock::now() < deadline) {
    for (auto* address = addresses.first; address; address = address->ai_next) {
      Socket candidate(::socket(address->ai_family, address->ai_socktype, address->ai_protocol));
      if (candidate.fd < 0) continue;
      int flags = ::fcntl(candidate.fd, F_GETFL, 0);
      if (flags < 0 || ::fcntl(candidate.fd, F_SETFL, flags | O_NONBLOCK) != 0) continue;
      int status = ::connect(candidate.fd, address->ai_addr, address->ai_addrlen);
      if (status != 0 && errno == EINPROGRESS) {
        fd_set ready;
        FD_ZERO(&ready);
        FD_SET(candidate.fd, &ready);
        timeval timeout{1, 0};
        status = ::select(candidate.fd + 1, nullptr, &ready, nullptr, &timeout);
        if (status > 0) {
          int socket_error = 0;
          socklen_t length = sizeof(socket_error);
          status = ::getsockopt(candidate.fd, SOL_SOCKET, SO_ERROR, &socket_error, &length);
          if (status == 0 && socket_error != 0) status = -1;
        } else status = -1;
      }
      if (status == 0 && ::fcntl(candidate.fd, F_SETFL, flags) == 0) {
        configure_socket(candidate.fd);
        return candidate;
      }
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  throw std::runtime_error("timed out connecting to TCP coordinator");
}

struct Handle {
  Socket peer;
  int rank;
  uint64_t sequence = 0;
};

void exchange_header(Handle& handle, uint64_t operation, uint64_t count) {
  uint64_t header[4] = {protocol_magic, ++handle.sequence, operation, count};
  uint64_t remote[4]{};
  send_all(handle.peer.fd, header, sizeof(header));
  recv_all(handle.peer.fd, remote, sizeof(remote));
  if (std::memcmp(header, remote, sizeof(header)) != 0)
    throw std::runtime_error("TCP collective sequence, operation, or size mismatch");
}
} // namespace

extern "C" {
const char* antfly_jaccl_last_error() noexcept { return last_error; }

int antfly_jaccl_open(int rank, const char* coordinator,
                      const char* /*devices_file*/, void** output) noexcept {
  if (!output || !coordinator || rank < 0 || rank > 1)
    return checked([] { throw std::invalid_argument("invalid TCP open arguments"); });
  *output = nullptr;
  return checked([&] {
    Socket peer = connect_peer(rank, coordinator);
    uint64_t hello[2] = {protocol_magic, static_cast<uint64_t>(rank)};
    uint64_t remote[2]{};
    send_all(peer.fd, hello, sizeof(hello));
    recv_all(peer.fd, remote, sizeof(remote));
    if (remote[0] != protocol_magic || remote[1] != static_cast<uint64_t>(1 - rank))
      throw std::runtime_error("TCP peer protocol or rank mismatch");
    *output = new Handle{std::move(peer), rank};
  });
}

int antfly_jaccl_rank(void* handle) noexcept {
  return handle ? static_cast<Handle*>(handle)->rank : -1;
}
int antfly_jaccl_size(void* handle) noexcept { return handle ? 2 : -1; }

int antfly_jaccl_all_sum_f32(void* handle, const float* input,
                             float* output, size_t count) noexcept {
  if (!handle || !input || !output || !count || count > max_payload / sizeof(float))
    return checked([] { throw std::invalid_argument("invalid TCP sum buffers"); });
  return checked([&] {
    auto& group = *static_cast<Handle*>(handle);
    exchange_header(group, 1, count);
    const size_t bytes = count * sizeof(float);
    if (group.rank == 0) {
      std::vector<float> peer(count);
      recv_all(group.peer.fd, peer.data(), bytes);
      for (size_t i = 0; i < count; ++i) output[i] = input[i] + peer[i];
      send_all(group.peer.fd, output, bytes);
    } else {
      send_all(group.peer.fd, input, bytes);
      recv_all(group.peer.fd, output, bytes);
    }
  });
}

int antfly_jaccl_all_gather(void* handle, const void* input, void* output,
                            size_t n_bytes) noexcept {
  if (!handle || !input || !output || !n_bytes || n_bytes > max_payload)
    return checked([] { throw std::invalid_argument("invalid TCP gather buffers"); });
  return checked([&] {
    auto& group = *static_cast<Handle*>(handle);
    exchange_header(group, 2, n_bytes);
    auto* gathered = static_cast<uint8_t*>(output);
    std::memcpy(gathered + group.rank * n_bytes, input, n_bytes);
    if (group.rank == 0) {
      recv_all(group.peer.fd, gathered + n_bytes, n_bytes);
      send_all(group.peer.fd, input, n_bytes);
    } else {
      send_all(group.peer.fd, input, n_bytes);
      recv_all(group.peer.fd, gathered, n_bytes);
    }
  });
}

int antfly_jaccl_barrier(void* handle) noexcept {
  if (!handle) return -1;
  return checked([&] { exchange_header(*static_cast<Handle*>(handle), 3, 0); });
}
void antfly_jaccl_close(void* handle) noexcept { delete static_cast<Handle*>(handle); }
} // extern "C"
