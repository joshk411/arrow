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

// Experimental (async-examples): accelerator offload hooks for the Parquet
// writer's page-construction path.
//
//  * DSA copy offload for the PLAIN encoder value copies and the page
//    concatenation copy (ColumnWriterImpl::ConcatenateBuffers).
//  * IAA (via QPL) DEFLATE/GZIP page compression, either synchronous (the
//    page writer blocks on every page) or asynchronous (the column writer
//    keeps encoding the next pages while IAA compresses earlier ones, and
//    writes completed pages out in order).
//
// Configuration (process-wide, read once from the environment, or set with
// SetConfig()):
//   PARQUET_DSA=off|sync|async     PARQUET_DSA_MIN=<bytes> (default 16384)
//   PARQUET_IAA=off|sync|async     PARQUET_IAA_DEPTH=<pages in flight> (8)
//   PARQUET_IAA_HUFFMAN=dynamic|fixed
//   PARQUET_IAA_PROGRESS=all|front    (progress-engine ablation)
//   PARQUET_CPU_ASYNC_THREADS=<n>     (0 = off) run the *CPU* gzip codec on n
//       worker threads through the same page pipeline. This is the "async-cpu"
//       control: same restructuring as the IAA async path, no accelerator, but
//       it costs n extra cores.
// IAA only applies to columns using Compression::GZIP.

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "parquet/platform.h"

namespace parquet::accel {

enum class Mode { kOff = 0, kSync = 1, kAsync = 2 };

struct Config {
  Mode dsa = Mode::kOff;
  int64_t dsa_min_bytes = 16384;
  Mode iaa = Mode::kOff;
  int iaa_depth = 8;
  bool iaa_dynamic_huffman = true;
  // true: polling advances all in-flight IAA jobs; false: only the awaited one.
  bool iaa_progress_all = true;
  // Admission heuristic: pages smaller than this compress on the CPU instead,
  // because IAA submission latency dominates on tiny buffers.
  int64_t iaa_min_bytes = 0;
  // >0: compress pages on this many CPU worker threads (async-cpu control).
  int cpu_async_threads = 0;
  int gzip_level = 1;
  // First core handed to the CPU worker pool; workers take base, base+1, ...
  int cpu_async_base = -1;
  // Explicit cores for the pool workers (worker i -> cpus[i]); overrides base.
  std::vector<int> cpu_async_cpus;
  // Pages each column writer may have queued on the pool (0 = cpu_async_threads).
  // With many writers sharing one pool, each needs only its share in flight.
  int cpu_async_depth = 0;
};

PARQUET_EXPORT const Config& GetConfig();
PARQUET_EXPORT void SetConfig(const Config& config);

struct Stats {
  uint64_t dsa_descriptors = 0;
  uint64_t dsa_bytes = 0;
  uint64_t dsa_partial = 0;
  uint64_t iaa_jobs = 0;
  uint64_t iaa_in_bytes = 0;
  uint64_t iaa_out_bytes = 0;
  uint64_t iaa_fallbacks = 0;  // jobs that failed / overflowed -> CPU codec
  uint64_t iaa_retries = 0;    // submissions rejected with QUEUES_ARE_BUSY, retried
};
// Per-thread counters.
PARQUET_EXPORT Stats GetStats();
PARQUET_EXPORT void ResetStats();

// ---------------------------------------------------------------- DSA copies
// Copies n bytes. Off / below threshold: memcpy. Sync: DSA, waits. Async:
// DSA, returns immediately; the caller must DrainCopies() before reading
// dst, or before src/dst may be freed, reused or reallocated.
PARQUET_EXPORT void Copy(void* dst, const void* src, size_t n);
PARQUET_EXPORT void DrainCopies();

// ---------------------------------------------------------- IAA compression
class IaaJob;

// Destination capacity IaaSubmit requires. QPL's HW dynamic-Huffman path can
// emit a corrupt stream (instead of an overflow error) when the output lands
// close to a tight bound such as zlib's compressBound, so callers must
// provide this much headroom.
inline int64_t IaaMinCapacity(int64_t n) { return n + (n >> 5) + 1024; }

// Submits a GZIP compression of [src, src+n) into [dst, dst+cap). Returns
// nullptr if IAA is unavailable or cap < IaaMinCapacity(n) (caller uses the
// CPU codec).
PARQUET_EXPORT IaaJob* IaaSubmit(const uint8_t* src, int64_t n, uint8_t* dst,
                                 int64_t cap);
// Non-blocking completion check.
PARQUET_EXPORT bool IaaDone(IaaJob* job);
// Waits, releases the job, returns the compressed size or -1 on failure
// (e.g. incompressible input overflowed dst); on -1 the caller must fall back.
PARQUET_EXPORT int64_t IaaFinish(IaaJob* job);
// Blocking convenience wrapper: -1 on failure.
PARQUET_EXPORT int64_t IaaCompress(const uint8_t* src, int64_t n, uint8_t* dst,
                                   int64_t cap);
PARQUET_EXPORT void CountIaaFallback();

// ------------------------------------------- CPU worker-thread compression
// Same submit/poll/finish shape as the IAA job API, but the work runs on a
// pool of CPU threads using Arrow's own GZIP codec, so the bytes produced are
// identical to the synchronous CPU path. Used only for the async-cpu control.
class CpuJob;
// Returns nullptr if the pool is disabled (cpu_async_threads == 0).
PARQUET_EXPORT CpuJob* CpuSubmit(const uint8_t* src, int64_t n, uint8_t* dst,
                                 int64_t cap);
PARQUET_EXPORT bool CpuDone(CpuJob* job);
// Waits, releases the job, returns compressed size or -1 on failure.
PARQUET_EXPORT int64_t CpuFinish(CpuJob* job);

}  // namespace parquet::accel
