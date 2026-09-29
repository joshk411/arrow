// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#include "parquet/accel_offload.h"

#include <algorithm>
#include <pthread.h>
#include <sched.h>
#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include <immintrin.h>

#include "arrow/util/compression.h"

#include "dsa_async.h"  // async-examples/common
#include "qpl/qpl.h"

namespace parquet::accel {

namespace {

Mode ParseMode(const char* s) {
  if (s == nullptr) return Mode::kOff;
  if (strcmp(s, "sync") == 0) return Mode::kSync;
  if (strcmp(s, "async") == 0) return Mode::kAsync;
  return Mode::kOff;
}

Config ConfigFromEnv() {
  Config c;
  c.dsa = ParseMode(getenv("PARQUET_DSA"));
  if (const char* v = getenv("PARQUET_DSA_MIN")) c.dsa_min_bytes = atoll(v);
  c.iaa = ParseMode(getenv("PARQUET_IAA"));
  if (const char* v = getenv("PARQUET_IAA_DEPTH")) c.iaa_depth = std::max(1, atoi(v));
  if (const char* v = getenv("PARQUET_IAA_HUFFMAN")) {
    c.iaa_dynamic_huffman = strcmp(v, "fixed") != 0;
  }
  if (const char* v = getenv("PARQUET_IAA_PROGRESS")) {
    c.iaa_progress_all = strcmp(v, "front") != 0;
  }
  if (const char* v = getenv("PARQUET_IAA_MIN")) c.iaa_min_bytes = atoll(v);
  if (const char* v = getenv("PARQUET_CPU_ASYNC_THREADS")) {
    c.cpu_async_threads = std::max(0, atoi(v));
  }
  return c;
}

Config& MutableConfig() {
  static Config config = ConfigFromEnv();
  return config;
}

thread_local Stats tl_stats;

// DSA writes whole 64B destination lines only; the CPU copies the unaligned
// head/tail so the device never shares a line with concurrent CPU stores.
constexpr uintptr_t kAlign = 64;

}  // namespace

const Config& GetConfig() { return MutableConfig(); }
void SetConfig(const Config& config) { MutableConfig() = config; }

Stats GetStats() {
  Stats s = tl_stats;
  if (auto* e = dsa_async::Engine::Get()) {
    s.dsa_descriptors = e->stats().submitted;
    s.dsa_bytes = e->stats().bytes;
    s.dsa_partial = e->stats().partial;
  }
  return s;
}

void ResetStats() {
  tl_stats = Stats();
  if (auto* e = dsa_async::Engine::Get()) e->ResetStats();
}

// ---------------------------------------------------------------- DSA copies

void Copy(void* dst, const void* src, size_t n) {
  const Config& c = GetConfig();
  if (c.dsa == Mode::kOff || static_cast<int64_t>(n) < c.dsa_min_bytes) {
    memcpy(dst, src, n);
    return;
  }
  auto* engine = dsa_async::Engine::Get();
  auto* d = static_cast<uint8_t*>(dst);
  auto* s = static_cast<const uint8_t*>(src);
  uintptr_t lo = (reinterpret_cast<uintptr_t>(d) + kAlign - 1) & ~(kAlign - 1);
  uintptr_t hi = (reinterpret_cast<uintptr_t>(d) + n) & ~(kAlign - 1);
  if (engine == nullptr || hi <= lo) {
    memcpy(dst, src, n);
    return;
  }
  size_t head = lo - reinterpret_cast<uintptr_t>(d);
  size_t body = hi - lo;
  uint64_t t = engine->Submit(d + head, s + head, body);
  memcpy(d, s, head);
  memcpy(d + head + body, s + head + body, n - head - body);
  if (c.dsa == Mode::kSync) engine->Wait(t);
}

void DrainCopies() {
  if (GetConfig().dsa != Mode::kAsync) return;
  if (auto* e = dsa_async::Engine::Get()) e->WaitAll();
}

// ---------------------------------------------------------- IAA compression

class IaaJob {
 public:
  bool Init() {
    uint32_t size = 0;
    if (qpl_get_job_size(qpl_path_hardware, &size) != QPL_STS_OK) return false;
    storage_ = std::make_unique<uint8_t[]>(size);
    job_ = reinterpret_cast<qpl_job*>(storage_.get());
    return qpl_init_job(qpl_path_hardware, job_) == QPL_STS_OK;
  }
  ~IaaJob() {
    if (job_ != nullptr) qpl_fini_job(job_);
  }

  qpl_job* job_ = nullptr;
  std::unique_ptr<uint8_t[]> storage_;
  int64_t in_size_ = 0;
  const uint8_t* src_ = nullptr;
  uint8_t* dst_ = nullptr;
  int64_t cap_ = 0;
  int credit_ = -1;  // socket whose in-flight credit this job holds
  qpl_status status_ = QPL_STS_OK;
  bool done_ = false;
};

namespace {

struct IaaPool {
  std::vector<std::unique_ptr<IaaJob>> owned;
  std::vector<IaaJob*> free_list;
  // Submitted, not yet observed complete. QPL's dynamic-Huffman compress is a
  // multi-stage job (statistics pass, then compress pass) and only advances to
  // the next stage inside qpl_check_job, so every in-flight job must be polled
  // (a progress engine), not just the one being waited on.
  std::vector<IaaJob*> inflight;
  bool unavailable = false;

  IaaJob* Acquire() {
    if (!free_list.empty()) {
      IaaJob* j = free_list.back();
      free_list.pop_back();
      return j;
    }
    if (unavailable) return nullptr;
    auto j = std::make_unique<IaaJob>();
    if (!j->Init()) {
      unavailable = true;
      return nullptr;
    }
    owned.push_back(std::move(j));
    return owned.back().get();
  }
  void Release(IaaJob* j) { free_list.push_back(j); }
};

IaaPool& Pool() {
  static thread_local IaaPool pool;
  return pool;
}

void Progress();

// Flow control: QPL spreads a thread's jobs over the IAA devices on its socket,
// each with a 64-entry shared WQ. With many async writers the WQs overflow and
// every rejected dynamic-Huffman job must restart from its first pass, so cap
// the jobs in flight per socket below the WQ capacity (PARQUET_IAA_INFLIGHT).
int InflightLimit() {
  static const int limit = [] {
    const char* v = getenv("PARQUET_IAA_INFLIGHT");
    return v ? atoi(v) : 192;
  }();
  return limit;
}
std::atomic<int> g_inflight[16];

int MySocket() {
  static thread_local int socket = -1;
  static thread_local int cached_cpu = -1;
  int cpu = sched_getcpu();
  if (cpu != cached_cpu) {
    cached_cpu = cpu;
    socket = 0;
    char path[96];
    snprintf(path, sizeof(path),
             "/sys/devices/system/cpu/cpu%d/topology/physical_package_id", cpu);
    if (FILE* f = fopen(path, "r")) {
      if (fscanf(f, "%d", &socket) != 1) socket = 0;
      fclose(f);
    }
    socket = std::clamp(socket, 0, 15);
  }
  return socket;
}

void AcquireCredit(IaaJob* j) {
  const int limit = InflightLimit();
  if (limit <= 0) return;
  const int s = MySocket();
  while (g_inflight[s].fetch_add(1, std::memory_order_acq_rel) >= limit) {
    g_inflight[s].fetch_sub(1, std::memory_order_acq_rel);
    Progress();
    _mm_pause();
  }
  j->credit_ = s;
}

void ReleaseCredit(IaaJob* j) {
  if (j->credit_ >= 0) {
    g_inflight[j->credit_].fetch_sub(1, std::memory_order_acq_rel);
    j->credit_ = -1;
  }
}

void PrepJob(IaaJob* j) {
  qpl_job* q = j->job_;
  q->op = qpl_op_compress;
  q->level = qpl_default_level;
  q->next_in_ptr = const_cast<uint8_t*>(j->src_);
  q->available_in = static_cast<uint32_t>(j->in_size_);
  q->next_out_ptr = j->dst_;
  q->available_out = static_cast<uint32_t>(j->cap_);
  q->total_in = 0;
  q->total_out = 0;
  q->flags = QPL_FLAG_FIRST | QPL_FLAG_LAST | QPL_FLAG_GZIP_MODE | QPL_FLAG_OMIT_VERIFY |
             (GetConfig().iaa_dynamic_huffman ? QPL_FLAG_DYNAMIC_HUFFMAN : 0);
}

// Shared WQ full (many writers per device): back off and resubmit instead of
// falling back to the CPU codec, like ENQCMD retry on DSA. A rejected submit
// may already have advanced the job (e.g. gzip header), so re-prepare it.
void SubmitWithRetry(IaaJob* j, bool progress = true) {
  PrepJob(j);
  j->status_ = qpl_submit_job(j->job_);
  while (j->status_ == QPL_STS_QUEUES_ARE_BUSY_ERR) {
    tl_stats.iaa_retries++;
    if (progress) Progress();  // not when called from inside Progress()
    _mm_pause();
    PrepJob(j);
    j->status_ = qpl_submit_job(j->job_);
  }
}

}  // namespace

IaaJob* IaaSubmit(const uint8_t* src, int64_t n, uint8_t* dst, int64_t cap) {
  if (n <= 0 || cap < IaaMinCapacity(n) || cap > UINT32_MAX) return nullptr;
  IaaJob* j = Pool().Acquire();
  if (j == nullptr) return nullptr;
  // IAA reports (rather than resolves) page faults, so touch every destination
  // page first: a fresh buffer from a non-main glibc arena may be unbacked.
  for (int64_t off = 0; off < cap; off += 4096) {
    reinterpret_cast<volatile uint8_t*>(dst)[off] = 0;
  }
  j->in_size_ = n;
  j->src_ = src;
  j->dst_ = dst;
  j->cap_ = cap;
  j->done_ = false;
  AcquireCredit(j);
  SubmitWithRetry(j);
  if (j->status_ != QPL_STS_OK) {
    ReleaseCredit(j);
    j->done_ = true;  // IaaFinish reports the failure
  } else {
    Pool().inflight.push_back(j);
  }
  return j;
}

namespace {

bool PollOne(IaaJob* j) {
  if (j->done_) return true;
  qpl_status st = qpl_check_job(j->job_);
  if (st == QPL_STS_BEING_PROCESSED) return false;
  if (st == QPL_STS_QUEUES_ARE_BUSY_ERR) {
    // Dynamic Huffman is two descriptors (histogram, then compress); QPL submits
    // the second one from qpl_check_job, where the WQ may be full. Restart.
    tl_stats.iaa_retries++;
    SubmitWithRetry(j, /*progress=*/false);
    if (j->status_ == QPL_STS_OK) return false;
    st = j->status_;
  }
  ReleaseCredit(j);
  j->status_ = st;
  j->done_ = true;
  return true;
}

// Advances every in-flight job; drops completed ones from the list.
void Progress() {
  auto& v = Pool().inflight;
  size_t k = 0;
  for (IaaJob* j : v) {
    if (!PollOne(j)) v[k++] = j;
  }
  v.resize(k);
}

}  // namespace

bool IaaDone(IaaJob* j) {
  if (j->done_) return true;
  if (GetConfig().iaa_progress_all) Progress();
  return PollOne(j);
}

int64_t IaaFinish(IaaJob* j) {
  while (!IaaDone(j)) {
    // busy-poll: completion latency is a few hundred microseconds per page
  }
  auto& v = Pool().inflight;
  for (size_t i = 0; i < v.size(); i++) {
    if (v[i] == j) {
      v.erase(v.begin() + i);
      break;
    }
  }
  int64_t out = -1;
  if (j->status_ == QPL_STS_OK && j->job_->total_out >= 8) {
    out = j->job_->total_out;
    // QPL's HW gzip path leaves ISIZE (input size mod 2^32) as 0; fix it.
    uint32_t isize = static_cast<uint32_t>(j->in_size_);
    memcpy(j->dst_ + out - 4, &isize, 4);
    tl_stats.iaa_jobs++;
    tl_stats.iaa_in_bytes += j->in_size_;
    tl_stats.iaa_out_bytes += out;
  }
  Pool().Release(j);
  return out;
}

int64_t IaaCompress(const uint8_t* src, int64_t n, uint8_t* dst, int64_t cap) {
  IaaJob* j = IaaSubmit(src, n, dst, cap);
  if (j == nullptr) return -1;
  return IaaFinish(j);
}

void CountIaaFallback() { tl_stats.iaa_fallbacks++; }

// ------------------------------------------- CPU worker-thread compression

class CpuJob {
 public:
  const uint8_t* src = nullptr;
  int64_t n = 0;
  uint8_t* dst = nullptr;
  int64_t cap = 0;
  int64_t result = -1;
  std::atomic<bool> done{false};
};

namespace {

// Fixed pool of gzip workers. Each worker owns its Codec, because Arrow codecs
// are not safe to share across threads.
class CpuPool {
 public:
  static CpuPool& Instance() {
    static CpuPool pool;
    return pool;
  }

  CpuJob* Submit(const uint8_t* src, int64_t n, uint8_t* dst, int64_t cap) {
    const Config& cfg = GetConfig();
    if (cfg.cpu_async_threads <= 0) return nullptr;
    Start(cfg.cpu_async_threads, cfg.gzip_level, cfg.cpu_async_base,
          cfg.cpu_async_cpus);
    CpuJob* j = nullptr;
    {
      std::lock_guard<std::mutex> lk(mu_);
      if (!free_.empty()) {
        j = free_.back();
        free_.pop_back();
      } else {
        owned_.push_back(std::make_unique<CpuJob>());
        j = owned_.back().get();
      }
    }
    j->src = src;
    j->n = n;
    j->dst = dst;
    j->cap = cap;
    j->result = -1;
    j->done.store(false, std::memory_order_release);
    {
      std::lock_guard<std::mutex> lk(mu_);
      queue_.push_back(j);
    }
    cv_.notify_one();
    return j;
  }

  void Release(CpuJob* j) {
    std::lock_guard<std::mutex> lk(mu_);
    free_.push_back(j);
  }

  ~CpuPool() {
    {
      std::lock_guard<std::mutex> lk(mu_);
      stop_ = true;
    }
    cv_.notify_all();
    for (auto& t : workers_) {
      if (t.joinable()) t.join();
    }
  }

 private:
  void Start(int n, int level, int cpu_base, const std::vector<int>& cpus) {
    std::lock_guard<std::mutex> lk(start_mu_);
    if (!workers_.empty()) return;
    level_ = level;
    for (int i = 0; i < n; i++) {
      // Workers inherit the caller's affinity mask, so without this they all
      // land on the caller's core and the "parallel" pool serializes.
      const int cpu = !cpus.empty() ? cpus[i % cpus.size()]
                      : cpu_base >= 0 ? cpu_base + i : -1;
      workers_.emplace_back([this, cpu] { Worker(cpu); });
    }
  }

  void Worker(int cpu) {
    if (cpu >= 0) {
      cpu_set_t set;
      CPU_ZERO(&set);
      CPU_SET(cpu, &set);
      pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
    }
    auto maybe = ::arrow::util::Codec::Create(
        ::arrow::Compression::GZIP, ::arrow::util::CodecOptions(level_));
    std::unique_ptr<::arrow::util::Codec> codec =
        maybe.ok() ? std::move(maybe).ValueOrDie() : nullptr;
    for (;;) {
      CpuJob* j = nullptr;
      {
        std::unique_lock<std::mutex> lk(mu_);
        cv_.wait(lk, [this] { return stop_ || !queue_.empty(); });
        if (stop_ && queue_.empty()) return;
        j = queue_.front();
        queue_.pop_front();
      }
      int64_t out = -1;
      if (codec != nullptr) {
        auto r = codec->Compress(j->n, j->src, j->cap, j->dst);
        if (r.ok()) out = *r;
      }
      j->result = out;
      j->done.store(true, std::memory_order_release);
    }
  }

  std::mutex mu_;
  std::mutex start_mu_;
  std::condition_variable cv_;
  std::deque<CpuJob*> queue_;
  std::vector<CpuJob*> free_;
  std::vector<std::unique_ptr<CpuJob>> owned_;
  std::vector<std::thread> workers_;
  int level_ = 1;
  bool stop_ = false;
};

}  // namespace

CpuJob* CpuSubmit(const uint8_t* src, int64_t n, uint8_t* dst, int64_t cap) {
  return CpuPool::Instance().Submit(src, n, dst, cap);
}

bool CpuDone(CpuJob* j) { return j->done.load(std::memory_order_acquire); }

int64_t CpuFinish(CpuJob* j) {
  while (!CpuDone(j)) {
    std::this_thread::yield();
  }
  int64_t out = j->result;
  CpuPool::Instance().Release(j);
  return out;
}

}  // namespace parquet::accel
