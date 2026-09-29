// Parquet writer benchmark for the sync-vs-async accelerator offload
// experiment (see README.md in this directory).
//
// Writes an in-memory Arrow table to an in-memory Parquet file and reports
// write throughput (uncompressed Arrow bytes per second of wall time),
// compressed size and DSA/IAA counters. The first iteration of every
// configuration is read back and compared against the source table.
//
// --threads N runs N independent writers (each pinned to --cpu-list[i], each
// with its own table built after pinning, its own sink and its own IAA jobs);
// throughput is then the aggregate. --pool-cpus pins the async-cpu gzip
// workers, which are shared by all writers.
#include <malloc.h>
#include <pthread.h>
#include <sched.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <thread>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "arrow/api.h"
#include "arrow/io/memory.h"
#include "parquet/accel_offload.h"
#include "parquet/arrow/reader.h"
#include "parquet/arrow/writer.h"
#include "parquet/properties.h"

namespace accel = parquet::accel;

struct Args {
  std::string dsa = "off";    // off|sync|async
  int64_t dsa_min = 16384;
  std::string iaa = "off";    // off|sync|async
  int depth = 8;
  std::string huffman = "dynamic";
  std::string progress = "all";
  int cpu_async = 0;
  int64_t iaa_min = 0;
  int64_t rows_per_page = 0;
  std::string codec = "gzip";  // none|gzip|snappy|zstd|lz4
  int level = -1;              // codec level (-1 = Arrow default)
  bool dictionary = false;
  int64_t rows = 1 << 22;
  int int_cols = 3;
  int dbl_cols = 1;
  int str_cols = 0;
  int64_t card = 1 << 16;      // distinct values per int column
  int64_t page = 1 << 20;      // data page size
  int64_t batch = 1024;        // write_batch_size
  int64_t row_group = 1 << 20;
  int iters = 5;
  int cpu = 40;
  int threads = 1;
  std::vector<int> cpus;       // writer t -> cpus[t] (default cpu + t)
  std::vector<int> pool_cpus;  // async-cpu pool workers
  int cpu_depth = 0;           // pages per writer queued on the pool (0 = pool size)
};

static std::vector<int> ParseList(const std::string& v) {
  std::vector<int> out;
  for (size_t p = 0; p < v.size();) {
    size_t q = v.find(',', p);
    if (q == std::string::npos) q = v.size();
    out.push_back(atoi(v.substr(p, q - p).c_str()));
    p = q + 1;
  }
  return out;
}

static accel::Mode ParseMode(const std::string& s) {
  if (s == "sync") return accel::Mode::kSync;
  if (s == "async") return accel::Mode::kAsync;
  return accel::Mode::kOff;
}

static bool ParseArgs(int argc, char** argv, Args& a) {
  for (int i = 1; i < argc; i++) {
    std::string k = argv[i];
    if (i + 1 >= argc) return false;
    std::string v = argv[++i];
    if (k == "--dsa") a.dsa = v;
    else if (k == "--dsa-min") a.dsa_min = atoll(v.c_str());
    else if (k == "--iaa") a.iaa = v;
    else if (k == "--depth") a.depth = atoi(v.c_str());
    else if (k == "--huffman") a.huffman = v;
    else if (k == "--progress") a.progress = v;
    else if (k == "--cpu-async") a.cpu_async = std::stoi(v);
    else if (k == "--iaa-min") a.iaa_min = atoll(v.c_str());
    else if (k == "--rows-per-page") a.rows_per_page = atoll(v.c_str());
    else if (k == "--codec") a.codec = v;
    else if (k == "--level") a.level = atoi(v.c_str());
    else if (k == "--dictionary") a.dictionary = v == "1" || v == "on";
    else if (k == "--rows") a.rows = atoll(v.c_str());
    else if (k == "--int-cols") a.int_cols = atoi(v.c_str());
    else if (k == "--dbl-cols") a.dbl_cols = atoi(v.c_str());
    else if (k == "--str-cols") a.str_cols = atoi(v.c_str());
    else if (k == "--card") a.card = atoll(v.c_str());
    else if (k == "--page") a.page = atoll(v.c_str());
    else if (k == "--batch") a.batch = atoll(v.c_str());
    else if (k == "--row-group") a.row_group = atoll(v.c_str());
    else if (k == "--iters") a.iters = atoi(v.c_str());
    else if (k == "--cpu") a.cpu = atoi(v.c_str());
    else if (k == "--threads") a.threads = atoi(v.c_str());
    else if (k == "--cpu-list") a.cpus = ParseList(v);
    else if (k == "--pool-cpus") a.pool_cpus = ParseList(v);
    else if (k == "--cpu-depth") a.cpu_depth = atoi(v.c_str());
    else return false;
  }
  if (a.cpus.empty())
    for (int t = 0; t < a.threads; t++) a.cpus.push_back(a.cpu + t);
  return static_cast<int>(a.cpus.size()) >= a.threads;
}

static arrow::Compression::type Codec(const std::string& c) {
  if (c == "gzip") return arrow::Compression::GZIP;
  if (c == "snappy") return arrow::Compression::SNAPPY;
  if (c == "zstd") return arrow::Compression::ZSTD;
  if (c == "lz4") return arrow::Compression::LZ4;
  return arrow::Compression::UNCOMPRESSED;
}

// Synthetic "telemetry" table: int64 columns drawn from a bounded domain
// (sorted-ish ids, skewed metrics), float64 measurements rounded to 2 decimal
// places, optional short strings. Compressible like typical fact tables.
static std::shared_ptr<arrow::Table> MakeTable(const Args& a, int64_t* raw_bytes) {
  std::mt19937_64 rng(42);
  std::vector<std::shared_ptr<arrow::Field>> fields;
  std::vector<std::shared_ptr<arrow::Array>> arrays;
  *raw_bytes = 0;
  for (int c = 0; c < a.int_cols; c++) {
    arrow::Int64Builder b;
    (void)b.Reserve(a.rows);
    int64_t base = static_cast<int64_t>(rng() % 1000000);
    for (int64_t r = 0; r < a.rows; r++) {
      int64_t v = (c == 0) ? base + r / 16                                // id-like
                           : static_cast<int64_t>((rng() % a.card) * (rng() % 4 + 1));
      b.UnsafeAppend(v);
    }
    std::shared_ptr<arrow::Array> arr;
    (void)b.Finish(&arr);
    arrays.push_back(arr);
    fields.push_back(arrow::field("i" + std::to_string(c), arrow::int64(), false));
    *raw_bytes += a.rows * 8;
  }
  std::normal_distribution<double> nd(100.0, 15.0);
  for (int c = 0; c < a.dbl_cols; c++) {
    arrow::DoubleBuilder b;
    (void)b.Reserve(a.rows);
    for (int64_t r = 0; r < a.rows; r++) b.UnsafeAppend(std::round(nd(rng) * 100) / 100);
    std::shared_ptr<arrow::Array> arr;
    (void)b.Finish(&arr);
    arrays.push_back(arr);
    fields.push_back(arrow::field("d" + std::to_string(c), arrow::float64(), false));
    *raw_bytes += a.rows * 8;
  }
  for (int c = 0; c < a.str_cols; c++) {
    arrow::StringBuilder b;
    for (int64_t r = 0; r < a.rows; r++) {
      std::string s = "host-" + std::to_string(rng() % a.card) + ".region-" +
                      std::to_string(rng() % 8);
      (void)b.Append(s);
      *raw_bytes += static_cast<int64_t>(s.size());
    }
    std::shared_ptr<arrow::Array> arr;
    (void)b.Finish(&arr);
    arrays.push_back(arr);
    fields.push_back(arrow::field("s" + std::to_string(c), arrow::utf8(), false));
  }
  return arrow::Table::Make(arrow::schema(fields), arrays);
}

static std::shared_ptr<parquet::WriterProperties> Props(const Args& a) {
  parquet::WriterProperties::Builder b;
  b.compression(Codec(a.codec));
  if (a.level >= 0) b.compression_level(a.level);
  if (a.dictionary) b.enable_dictionary();
  else b.disable_dictionary();
  b.data_pagesize(a.page);
  if (a.rows_per_page > 0) b.max_rows_per_page(a.rows_per_page);
  b.write_batch_size(a.batch);
  b.max_row_group_length(a.row_group);
  return b.build();
}

static void Pin(int cpu) {
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(cpu, &set);
  pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
}

struct WriterResult {
  std::vector<double> secs;  // per timed iteration
  int64_t raw = 0;
  int64_t file_size = 0;
  bool verified = false;
  accel::Stats st{};
  double t_end = 0;  // seconds after the start barrier
};

// One writer: builds its own table (node-local after pinning), then iteration 0
// (warm-up + round-trip verify), then waits for the start barrier and runs the
// timed iterations.
static void Writer(const Args& a, int tid, std::atomic<int>& ready,
                   std::atomic<bool>& go,
                   std::chrono::steady_clock::time_point* t0, WriterResult& res) {
  Pin(a.cpus[tid]);
  auto table = MakeTable(a, &res.raw);
  auto props = Props(a);
  auto arrow_props = parquet::ArrowWriterProperties::Builder().build();
  // One pre-faulted output buffer per writer, reused every iteration (a >64 MB
  // allocation from a non-main glibc arena is always a fresh mmap, so allocating
  // the sink per iteration would add ~40K page faults to every timed write).
  const int64_t cap = res.raw + (res.raw >> 2);
  std::shared_ptr<arrow::ResizableBuffer> out =
      arrow::AllocateResizableBuffer(cap).ValueOrDie();
  memset(out->mutable_data(), 0, cap);
  for (int it = 0; it < a.iters + 1; it++) {
    if (it == 1) {
      ready.fetch_add(1);
      while (!go.load(std::memory_order_acquire)) std::this_thread::yield();
    }
    if (!out->Resize(cap, false).ok()) return;
    auto sink = std::make_shared<arrow::io::BufferOutputStream>(out);
    accel::ResetStats();
    auto s0 = std::chrono::steady_clock::now();
    auto s = parquet::arrow::WriteTable(*table, arrow::default_memory_pool(), sink,
                                        a.row_group, props, arrow_props);
    auto buf = sink->Finish().ValueOrDie();
    auto s1 = std::chrono::steady_clock::now();
    if (!s.ok()) {
      fprintf(stderr, "write failed: %s\n", s.ToString().c_str());
      return;
    }
    res.file_size = buf->size();
    if (it == 0) {
      auto reader = parquet::arrow::OpenFile(std::make_shared<arrow::io::BufferReader>(buf),
                                             arrow::default_memory_pool())
                        .ValueOrDie();
      std::shared_ptr<arrow::Table> back;
      auto rs = reader->ReadTable(&back);
      res.verified = rs.ok() && back->Equals(*table);
      if (!res.verified) fprintf(stderr, "verify failed: %s\n", rs.ToString().c_str());
    } else {
      res.secs.push_back(std::chrono::duration<double>(s1 - s0).count());
      res.st = accel::GetStats();
    }
  }
  res.t_end = std::chrono::duration<double>(std::chrono::steady_clock::now() - *t0).count();
  // End barrier: hold the table and buffers until every writer is done. Freeing
  // ~300 MB per writer (munmap -> IOTLB invalidations for the SVM-attached IAA/DSA)
  // while others still run stalls the shared devices and makes stragglers.
  ready.fetch_sub(1);
  while (ready.load() > 0) std::this_thread::yield();
}

int main(int argc, char** argv) {
  Args a;
  if (!ParseArgs(argc, argv, a)) {
    fprintf(stderr, "bad args\n");
    return 2;
  }
  // Keep freed pages in the heap (like jemalloc/mimalloc would): avoids a
  // fresh mmap + page faults on every page buffer, for every mode alike.
  mallopt(M_MMAP_THRESHOLD, 1 << 30);
  mallopt(M_TRIM_THRESHOLD, 1 << 30);
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(a.cpu, &set);
  pthread_setaffinity_np(pthread_self(), sizeof(set), &set);

  accel::Config cfg;
  cfg.dsa = ParseMode(a.dsa);
  cfg.dsa_min_bytes = a.dsa_min;
  cfg.iaa = ParseMode(a.iaa);
  cfg.iaa_depth = a.depth;
  cfg.iaa_dynamic_huffman = a.huffman != "fixed";
  cfg.iaa_progress_all = a.progress != "front";
  cfg.cpu_async_threads = a.cpu_async;
  cfg.iaa_min_bytes = a.iaa_min;
  cfg.gzip_level = a.level > 0 ? a.level : 1;
  cfg.cpu_async_base = a.cpu + 1;
  cfg.cpu_async_cpus = a.pool_cpus;
  cfg.cpu_async_depth = a.cpu_depth;
  accel::SetConfig(cfg);

  std::vector<WriterResult> res(a.threads);
  std::atomic<int> ready{0};
  std::atomic<bool> go{false};
  std::chrono::steady_clock::time_point t0;
  std::vector<std::thread> ths;
  for (int t = 0; t < a.threads; t++)
    ths.emplace_back(Writer, std::cref(a), t, std::ref(ready), std::ref(go), &t0,
                     std::ref(res[t]));
  while (ready.load() < a.threads) std::this_thread::sleep_for(std::chrono::milliseconds(1));
  t0 = std::chrono::steady_clock::now();
  go.store(true, std::memory_order_release);
  for (auto& th : ths) th.join();

  // One writer: median iteration time (as before). Several: aggregate bytes
  // over the wall time from the start barrier to the last writer finishing.
  std::vector<double> secs = res[0].secs;
  const int64_t raw = res[0].raw;
  const int64_t file_size = res[0].file_size;
  bool verified = true;
  accel::Stats st{};
  double wall = 0;
  for (auto& r : res) {
    verified = verified && r.verified && static_cast<int>(r.secs.size()) == a.iters;
    wall = std::max(wall, r.t_end);
    st.dsa_descriptors += r.st.dsa_descriptors;
    st.dsa_bytes += r.st.dsa_bytes;
    st.dsa_partial += r.st.dsa_partial;
    st.iaa_jobs += r.st.iaa_jobs;
    st.iaa_fallbacks += r.st.iaa_fallbacks;
    st.iaa_retries += r.st.iaa_retries;
  }
  if (getenv("PQ_DEBUG")) {
    std::vector<double> ends;
    for (size_t i = 0; i < res.size(); i++) { ends.push_back(res[i].t_end); if (getenv("PQ_DEBUG")[0] == '2') { fprintf(stderr, "w %zu cpu %d end %.3f secs", i, a.cpus[i], res[i].t_end); for (double x : res[i].secs) fprintf(stderr, " %.3f", x); fprintf(stderr, "\n"); } }
    std::sort(ends.begin(), ends.end());
    fprintf(stderr, "t_end min %.3f p50 %.3f max %.3f\n", ends.front(), ends[ends.size() / 2],
            ends.back());
  }
  std::sort(secs.begin(), secs.end());
  double med = secs[secs.size() / 2];
  const double gbps = a.threads == 1 ? raw / med / 1e9
                                     : static_cast<double>(raw) * a.iters * a.threads / wall / 1e9;
  printf("dsa,dsa_min,iaa,iaa_min,depth,huffman,progress,cpu_async,codec,level,dictionary,rows,cols,page,batch,"
         "raw_bytes,file_bytes,ratio,secs_median,secs_min,gbps,dsa_descs,dsa_bytes,"
         "dsa_partial,iaa_jobs,iaa_fallbacks,verified,threads,pool,iaa_retries\n");
  printf("%s,%ld,%s,%ld,%d,%s,%s,%d,%s,%d,%d,%ld,%d,%ld,%ld,%ld,%ld,%.3f,%.5f,%.5f,%.4f,%lu,%lu,%lu,"
         "%lu,%lu,%d,%d,%zu,%lu\n",
         a.dsa.c_str(), a.dsa_min, a.iaa.c_str(), a.iaa_min, a.depth, a.huffman.c_str(), a.progress.c_str(), a.cpu_async,
         a.codec.c_str(), a.level, a.dictionary, a.rows,
         a.int_cols + a.dbl_cols + a.str_cols, a.page, a.batch, raw, file_size,
         static_cast<double>(raw) / file_size, med, secs[0], gbps,
         st.dsa_descriptors, st.dsa_bytes, st.dsa_partial, st.iaa_jobs, st.iaa_fallbacks,
         verified, a.threads,
         a.pool_cpus.empty() ? static_cast<size_t>(a.cpu_async) : a.pool_cpus.size(),
         st.iaa_retries);
  return verified ? 0 : 1;
}
