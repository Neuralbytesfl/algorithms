#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

using Clock = std::chrono::steady_clock;

enum class PoolPhase {
    Count,
    Scatter
};

struct Result {
    std::string name;
    std::string dataset;
    std::size_t size;
    double median_ms;
    bool sorted;
};

struct MemoryStatus {
    std::size_t rss_kb = 0;
    std::size_t hwm_kb = 0;
};

struct Record {
    std::int64_t key;
    std::uint32_t category;
    double score;
    std::array<char, 24> label;
};

static void insertion_sort_range(std::vector<int>& a, std::size_t lo, std::size_t hi) {
    for (std::size_t i = lo + 1; i < hi; ++i) {
        int value = a[i];
        std::size_t j = i;
        while (j > lo && a[j - 1] > value) {
            a[j] = a[j - 1];
            --j;
        }
        a[j] = value;
    }
}

static void merge_into(std::vector<int>& a, std::vector<int>& tmp,
                       std::size_t lo, std::size_t mid, std::size_t hi) {
    std::size_t i = lo;
    std::size_t j = mid;
    std::size_t out = lo;

    while (i < mid && j < hi) {
        if (a[j] < a[i]) {
            tmp[out++] = a[j++];
        } else {
            tmp[out++] = a[i++];
        }
    }
    while (i < mid) tmp[out++] = a[i++];
    while (j < hi) tmp[out++] = a[j++];

    std::copy(tmp.begin() + static_cast<std::ptrdiff_t>(lo),
              tmp.begin() + static_cast<std::ptrdiff_t>(hi),
              a.begin() + static_cast<std::ptrdiff_t>(lo));
}

static void plain_merge_sort_impl(std::vector<int>& a, std::vector<int>& tmp,
                                  std::size_t lo, std::size_t hi) {
    if (hi - lo <= 1) return;
    const std::size_t mid = lo + (hi - lo) / 2;
    plain_merge_sort_impl(a, tmp, lo, mid);
    plain_merge_sort_impl(a, tmp, mid, hi);
    merge_into(a, tmp, lo, mid, hi);
}

static void plain_merge_sort(std::vector<int>& a) {
    std::vector<int> tmp(a.size());
    plain_merge_sort_impl(a, tmp, 0, a.size());
}

static void tuned_merge_sort_impl(std::vector<int>& a, std::vector<int>& tmp,
                                  std::size_t lo, std::size_t hi) {
    constexpr std::size_t insertion_cutoff = 32;
    if (hi - lo <= insertion_cutoff) {
        insertion_sort_range(a, lo, hi);
        return;
    }

    const std::size_t mid = lo + (hi - lo) / 2;
    tuned_merge_sort_impl(a, tmp, lo, mid);
    tuned_merge_sort_impl(a, tmp, mid, hi);

    if (a[mid - 1] <= a[mid]) return;
    merge_into(a, tmp, lo, mid, hi);
}

static void tuned_merge_sort(std::vector<int>& a) {
    std::vector<int> tmp(a.size());
    tuned_merge_sort_impl(a, tmp, 0, a.size());
}

static void bottom_up_merge_sort(std::vector<int>& a) {
    constexpr std::size_t insertion_cutoff = 32;
    const std::size_t n = a.size();
    std::vector<int> tmp(n);

    for (std::size_t lo = 0; lo < n; lo += insertion_cutoff) {
        insertion_sort_range(a, lo, std::min(lo + insertion_cutoff, n));
    }

    for (std::size_t width = insertion_cutoff; width < n; width *= 2) {
        for (std::size_t lo = 0; lo < n; lo += width * 2) {
            const std::size_t mid = std::min(lo + width, n);
            const std::size_t hi = std::min(lo + width * 2, n);
            if (mid >= hi || a[mid - 1] <= a[mid]) continue;
            merge_into(a, tmp, lo, mid, hi);
        }
    }
}

static void radix_sort_i32(std::vector<int>& a) {
    constexpr std::size_t radix = 256;
    constexpr std::size_t passes = 4;
    std::vector<int> tmp(a.size());
    std::array<std::size_t, radix> count{};

    for (std::size_t pass = 0; pass < passes; ++pass) {
        count.fill(0);
        const unsigned shift = static_cast<unsigned>(pass * 8);

        for (int value : a) {
            const std::uint32_t key = static_cast<std::uint32_t>(value) ^ 0x80000000u;
            ++count[(key >> shift) & 0xFFu];
        }

        std::size_t sum = 0;
        for (std::size_t i = 0; i < radix; ++i) {
            const std::size_t next = sum + count[i];
            count[i] = sum;
            sum = next;
        }

        for (int value : a) {
            const std::uint32_t key = static_cast<std::uint32_t>(value) ^ 0x80000000u;
            tmp[count[(key >> shift) & 0xFFu]++] = value;
        }

        a.swap(tmp);
    }
}

static void radix_sort_i32_16bit(std::vector<int>& a) {
    constexpr std::size_t radix = 1u << 16u;
    std::vector<int> tmp(a.size());
    std::vector<std::size_t> count(radix);

    for (unsigned shift : {0u, 16u}) {
        std::fill(count.begin(), count.end(), 0);

        for (int value : a) {
            const std::uint32_t key = static_cast<std::uint32_t>(value) ^ 0x80000000u;
            ++count[(key >> shift) & 0xFFFFu];
        }

        std::size_t sum = 0;
        for (std::size_t i = 0; i < radix; ++i) {
            const std::size_t next = sum + count[i];
            count[i] = sum;
            sum = next;
        }

        for (int value : a) {
            const std::uint32_t key = static_cast<std::uint32_t>(value) ^ 0x80000000u;
            tmp[count[(key >> shift) & 0xFFFFu]++] = value;
        }

        a.swap(tmp);
    }
}

static void radix_sort_i32_16bit_fused_counts(std::vector<int>& a) {
    constexpr std::size_t radix = 1u << 16u;
    std::vector<int> tmp(a.size());
    std::vector<std::size_t> low_count(radix);
    std::vector<std::size_t> high_count(radix);

    for (int value : a) {
        const std::uint32_t key = static_cast<std::uint32_t>(value) ^ 0x80000000u;
        ++low_count[key & 0xFFFFu];
        ++high_count[key >> 16u];
    }

    std::size_t sum = 0;
    for (std::size_t i = 0; i < radix; ++i) {
        const std::size_t next = sum + low_count[i];
        low_count[i] = sum;
        sum = next;
    }

    for (int value : a) {
        const std::uint32_t key = static_cast<std::uint32_t>(value) ^ 0x80000000u;
        tmp[low_count[key & 0xFFFFu]++] = value;
    }

    sum = 0;
    for (std::size_t i = 0; i < radix; ++i) {
        const std::size_t next = sum + high_count[i];
        high_count[i] = sum;
        sum = next;
    }

    for (int value : tmp) {
        const std::uint32_t key = static_cast<std::uint32_t>(value) ^ 0x80000000u;
        a[high_count[key >> 16u]++] = value;
    }
}

static void radix_sort_i32_16bit_fused_reuse(std::vector<int>& a) {
    constexpr std::size_t radix = 1u << 16u;
    thread_local std::vector<int> tmp;
    thread_local std::vector<std::size_t> low_count;
    thread_local std::vector<std::size_t> high_count;

    tmp.resize(a.size());
    low_count.resize(radix);
    high_count.resize(radix);
    std::fill(low_count.begin(), low_count.end(), 0);
    std::fill(high_count.begin(), high_count.end(), 0);

    for (int value : a) {
        const std::uint32_t key = static_cast<std::uint32_t>(value) ^ 0x80000000u;
        ++low_count[key & 0xFFFFu];
        ++high_count[key >> 16u];
    }

    std::size_t sum = 0;
    for (std::size_t i = 0; i < radix; ++i) {
        const std::size_t next = sum + low_count[i];
        low_count[i] = sum;
        sum = next;
    }

    for (int value : a) {
        const std::uint32_t key = static_cast<std::uint32_t>(value) ^ 0x80000000u;
        tmp[low_count[key & 0xFFFFu]++] = value;
    }

    sum = 0;
    for (std::size_t i = 0; i < radix; ++i) {
        const std::size_t next = sum + high_count[i];
        high_count[i] = sum;
        sum = next;
    }

    for (int value : tmp) {
        const std::uint32_t key = static_cast<std::uint32_t>(value) ^ 0x80000000u;
        a[high_count[key >> 16u]++] = value;
    }
}

static void radix_pass_parallel(const std::vector<int>& in, std::vector<int>& out,
                                unsigned shift, unsigned thread_count) {
    constexpr std::size_t radix = 1u << 16u;
    const std::size_t n = in.size();
    std::vector<std::size_t> offsets(static_cast<std::size_t>(thread_count) * radix);
    std::vector<std::thread> workers;
    workers.reserve(thread_count);

    for (unsigned t = 0; t < thread_count; ++t) {
        workers.emplace_back([&, t] {
            const std::size_t lo = n * t / thread_count;
            const std::size_t hi = n * (t + 1) / thread_count;
            std::size_t* local = offsets.data() + static_cast<std::size_t>(t) * radix;
            for (std::size_t i = lo; i < hi; ++i) {
                const std::uint32_t key = static_cast<std::uint32_t>(in[i]) ^ 0x80000000u;
                ++local[(key >> shift) & 0xFFFFu];
            }
        });
    }
    for (std::thread& worker : workers) worker.join();

    std::size_t base = 0;
    for (std::size_t bucket = 0; bucket < radix; ++bucket) {
        std::size_t bucket_total = 0;
        for (unsigned t = 0; t < thread_count; ++t) {
            const std::size_t index = static_cast<std::size_t>(t) * radix + bucket;
            const std::size_t count = offsets[index];
            offsets[index] = base + bucket_total;
            bucket_total += count;
        }
        base += bucket_total;
    }

    workers.clear();
    for (unsigned t = 0; t < thread_count; ++t) {
        workers.emplace_back([&, t] {
            const std::size_t lo = n * t / thread_count;
            const std::size_t hi = n * (t + 1) / thread_count;
            std::size_t* local = offsets.data() + static_cast<std::size_t>(t) * radix;
            for (std::size_t i = lo; i < hi; ++i) {
                const int value = in[i];
                const std::uint32_t key = static_cast<std::uint32_t>(value) ^ 0x80000000u;
                out[local[(key >> shift) & 0xFFFFu]++] = value;
            }
        });
    }
    for (std::thread& worker : workers) worker.join();
}

static void radix_sort_i32_16bit_parallel(std::vector<int>& a) {
    const unsigned hardware = std::max(1u, std::thread::hardware_concurrency());
    const unsigned thread_count = std::min(8u, hardware);
    if (thread_count <= 1 || a.size() < 262144) {
        radix_sort_i32_16bit_fused_reuse(a);
        return;
    }

    thread_local std::vector<int> tmp;
    tmp.resize(a.size());
    radix_pass_parallel(a, tmp, 0u, thread_count);
    radix_pass_parallel(tmp, a, 16u, thread_count);
}

static void radix_sort_i32_11bit_fused_counts(std::vector<int>& a) {
    constexpr std::size_t radix11 = 1u << 11u;
    constexpr std::size_t radix10 = 1u << 10u;
    thread_local std::vector<int> tmp;
    thread_local std::vector<std::size_t> count0;
    thread_local std::vector<std::size_t> count1;
    thread_local std::vector<std::size_t> count2;

    tmp.resize(a.size());
    count0.assign(radix11, 0);
    count1.assign(radix11, 0);
    count2.assign(radix10, 0);

    for (int value : a) {
        const std::uint32_t key = static_cast<std::uint32_t>(value) ^ 0x80000000u;
        ++count0[key & 0x7FFu];
        ++count1[(key >> 11u) & 0x7FFu];
        ++count2[key >> 22u];
    }

    auto prefix = [](std::vector<std::size_t>& count) {
        std::size_t sum = 0;
        for (std::size_t& item : count) {
            const std::size_t next = sum + item;
            item = sum;
            sum = next;
        }
    };

    prefix(count0);
    for (int value : a) {
        const std::uint32_t key = static_cast<std::uint32_t>(value) ^ 0x80000000u;
        tmp[count0[key & 0x7FFu]++] = value;
    }

    prefix(count1);
    for (int value : tmp) {
        const std::uint32_t key = static_cast<std::uint32_t>(value) ^ 0x80000000u;
        a[count1[(key >> 11u) & 0x7FFu]++] = value;
    }

    prefix(count2);
    for (int value : a) {
        const std::uint32_t key = static_cast<std::uint32_t>(value) ^ 0x80000000u;
        tmp[count2[key >> 22u]++] = value;
    }

    a.swap(tmp);
}

static void radix_sort_i32_16bit_openmp(std::vector<int>& a) {
#ifndef _OPENMP
    radix_sort_i32_16bit_fused_reuse(a);
#else
    constexpr std::size_t radix = 1u << 16u;
    const int thread_count = std::min(8, std::max(1, omp_get_max_threads()));
    const std::size_t n = a.size();
    std::vector<int> tmp(n);
    std::vector<std::size_t> offsets(static_cast<std::size_t>(thread_count) * radix);

    auto run_pass = [&](const std::vector<int>& in, std::vector<int>& out, unsigned shift) {
        std::fill(offsets.begin(), offsets.end(), 0);

#pragma omp parallel num_threads(thread_count)
        {
            const int t = omp_get_thread_num();
            const std::size_t lo = n * static_cast<std::size_t>(t) / static_cast<std::size_t>(thread_count);
            const std::size_t hi = n * static_cast<std::size_t>(t + 1) / static_cast<std::size_t>(thread_count);
            std::size_t* local = offsets.data() + static_cast<std::size_t>(t) * radix;
            for (std::size_t i = lo; i < hi; ++i) {
                const std::uint32_t key = static_cast<std::uint32_t>(in[i]) ^ 0x80000000u;
                ++local[(key >> shift) & 0xFFFFu];
            }
        }

        std::size_t base = 0;
        for (std::size_t bucket = 0; bucket < radix; ++bucket) {
            std::size_t bucket_total = 0;
            for (int t = 0; t < thread_count; ++t) {
                const std::size_t index = static_cast<std::size_t>(t) * radix + bucket;
                const std::size_t count = offsets[index];
                offsets[index] = base + bucket_total;
                bucket_total += count;
            }
            base += bucket_total;
        }

#pragma omp parallel num_threads(thread_count)
        {
            const int t = omp_get_thread_num();
            const std::size_t lo = n * static_cast<std::size_t>(t) / static_cast<std::size_t>(thread_count);
            const std::size_t hi = n * static_cast<std::size_t>(t + 1) / static_cast<std::size_t>(thread_count);
            std::size_t* local = offsets.data() + static_cast<std::size_t>(t) * radix;
            for (std::size_t i = lo; i < hi; ++i) {
                const int value = in[i];
                const std::uint32_t key = static_cast<std::uint32_t>(value) ^ 0x80000000u;
                out[local[(key >> shift) & 0xFFFFu]++] = value;
            }
        }
    };

    run_pass(a, tmp, 0u);
    run_pass(tmp, a, 16u);
#endif
}

class Radix16WorkerPool {
public:
    explicit Radix16WorkerPool(unsigned requested_threads)
        : thread_count_(std::max(1u, requested_threads)),
          offsets_(static_cast<std::size_t>(thread_count_) * radix_) {
        workers_.reserve(thread_count_);
        for (unsigned id = 0; id < thread_count_; ++id) {
            workers_.emplace_back([this, id] { worker_loop(id); });
        }
    }

    ~Radix16WorkerPool() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
            ++generation_;
        }
        start_cv_.notify_all();
        for (std::thread& worker : workers_) {
            if (worker.joinable()) worker.join();
        }
    }

    void sort(std::vector<int>& a) {
        if (thread_count_ <= 1 || a.size() < 262144) {
            radix_sort_i32_16bit_fused_reuse(a);
            return;
        }

        tmp_.resize(a.size());
        offsets_.assign(static_cast<std::size_t>(thread_count_) * radix_, 0);
        run_phase(PoolPhase::Count, &a, &tmp_, 0u);
        prefix_offsets();
        run_phase(PoolPhase::Scatter, &a, &tmp_, 0u);

        std::fill(offsets_.begin(), offsets_.end(), 0);
        run_phase(PoolPhase::Count, &tmp_, &a, 16u);
        prefix_offsets();
        run_phase(PoolPhase::Scatter, &tmp_, &a, 16u);
    }

private:
    static constexpr std::size_t radix_ = 1u << 16u;

    void run_phase(PoolPhase phase, const std::vector<int>* input,
                   std::vector<int>* output, unsigned shift) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            phase_ = phase;
            input_ = input;
            output_ = output;
            shift_ = shift;
            active_workers_ = thread_count_;
            ++generation_;
        }
        start_cv_.notify_all();

        std::unique_lock<std::mutex> lock(mutex_);
        done_cv_.wait(lock, [&] { return active_workers_ == 0; });
    }

    void prefix_offsets() {
        std::size_t base = 0;
        for (std::size_t bucket = 0; bucket < radix_; ++bucket) {
            std::size_t bucket_total = 0;
            for (unsigned t = 0; t < thread_count_; ++t) {
                const std::size_t index = static_cast<std::size_t>(t) * radix_ + bucket;
                const std::size_t count = offsets_[index];
                offsets_[index] = base + bucket_total;
                bucket_total += count;
            }
            base += bucket_total;
        }
    }

    void worker_loop(unsigned id) {
        std::size_t seen_generation = 0;
        while (true) {
            PoolPhase phase;
            const std::vector<int>* input;
            std::vector<int>* output;
            unsigned shift;
            std::size_t n;

            {
                std::unique_lock<std::mutex> lock(mutex_);
                start_cv_.wait(lock, [&] { return stopping_ || generation_ != seen_generation; });
                if (stopping_) return;
                seen_generation = generation_;
                phase = phase_;
                input = input_;
                output = output_;
                shift = shift_;
                n = input_->size();
            }

            const std::size_t lo = n * id / thread_count_;
            const std::size_t hi = n * (id + 1) / thread_count_;
            std::size_t* local = offsets_.data() + static_cast<std::size_t>(id) * radix_;

            if (phase == PoolPhase::Count) {
                for (std::size_t i = lo; i < hi; ++i) {
                    const std::uint32_t key = static_cast<std::uint32_t>((*input)[i]) ^ 0x80000000u;
                    ++local[(key >> shift) & 0xFFFFu];
                }
            } else {
                for (std::size_t i = lo; i < hi; ++i) {
                    const int value = (*input)[i];
                    const std::uint32_t key = static_cast<std::uint32_t>(value) ^ 0x80000000u;
                    (*output)[local[(key >> shift) & 0xFFFFu]++] = value;
                }
            }

            {
                std::lock_guard<std::mutex> lock(mutex_);
                --active_workers_;
                if (active_workers_ == 0) done_cv_.notify_one();
            }
        }
    }

    unsigned thread_count_;
    std::vector<std::thread> workers_;
    std::vector<int> tmp_;
    std::vector<std::size_t> offsets_;
    std::mutex mutex_;
    std::condition_variable start_cv_;
    std::condition_variable done_cv_;
    bool stopping_ = false;
    std::size_t generation_ = 0;
    unsigned active_workers_ = 0;
    PoolPhase phase_ = PoolPhase::Count;
    const std::vector<int>* input_ = nullptr;
    std::vector<int>* output_ = nullptr;
    unsigned shift_ = 0;
};

static void radix_sort_i32_16bit_pool(std::vector<int>& a) {
    static Radix16WorkerPool pool(std::min(8u, std::max(1u, std::thread::hardware_concurrency())));
    pool.sort(a);
}

static void radix_sort_i32_16bit_pool2(std::vector<int>& a) {
    static Radix16WorkerPool pool(2);
    pool.sort(a);
}

static void radix_sort_i32_16bit_pool4(std::vector<int>& a) {
    static Radix16WorkerPool pool(4);
    pool.sort(a);
}

static void radix_sort_i64_16bit(std::vector<std::int64_t>& a) {
    constexpr std::size_t radix = 1u << 16u;
    thread_local std::vector<std::int64_t> tmp;
    thread_local std::vector<std::size_t> count;

    tmp.resize(a.size());
    count.resize(radix);

    for (unsigned shift : {0u, 16u, 32u, 48u}) {
        std::fill(count.begin(), count.end(), 0);

        for (std::int64_t value : a) {
            const std::uint64_t key = static_cast<std::uint64_t>(value) ^ 0x8000000000000000ull;
            ++count[(key >> shift) & 0xFFFFull];
        }

        std::size_t sum = 0;
        for (std::size_t i = 0; i < radix; ++i) {
            const std::size_t next = sum + count[i];
            count[i] = sum;
            sum = next;
        }

        for (std::int64_t value : a) {
            const std::uint64_t key = static_cast<std::uint64_t>(value) ^ 0x8000000000000000ull;
            tmp[count[(key >> shift) & 0xFFFFull]++] = value;
        }

        a.swap(tmp);
    }
}

static void adaptive_i64_sort(std::vector<std::int64_t>& a) {
    if (a.size() < 2) return;

    bool ascending = true;
    bool descending = true;
    for (std::size_t i = 1; i < a.size(); ++i) {
        ascending = ascending && a[i - 1] <= a[i];
        descending = descending && a[i - 1] >= a[i];
    }

    if (ascending) return;
    if (descending) {
        std::reverse(a.begin(), a.end());
        return;
    }

    radix_sort_i64_16bit(a);
}

static void radix_sort_records_by_key(std::vector<Record>& a) {
    constexpr std::size_t radix = 1u << 16u;
    thread_local std::vector<Record> tmp;
    thread_local std::vector<std::size_t> count;

    tmp.resize(a.size());
    count.resize(radix);

    for (unsigned shift : {0u, 16u, 32u, 48u}) {
        std::fill(count.begin(), count.end(), 0);

        for (const Record& value : a) {
            const std::uint64_t key = static_cast<std::uint64_t>(value.key) ^ 0x8000000000000000ull;
            ++count[(key >> shift) & 0xFFFFull];
        }

        std::size_t sum = 0;
        for (std::size_t i = 0; i < radix; ++i) {
            const std::size_t next = sum + count[i];
            count[i] = sum;
            sum = next;
        }

        for (const Record& value : a) {
            const std::uint64_t key = static_cast<std::uint64_t>(value.key) ^ 0x8000000000000000ull;
            tmp[count[(key >> shift) & 0xFFFFull]++] = value;
        }

        a.swap(tmp);
    }
}

static void counting_sort_known_range(std::vector<int>& a, int min_value, int max_value) {
    const std::uint64_t range = static_cast<std::uint64_t>(
        static_cast<std::int64_t>(max_value) - static_cast<std::int64_t>(min_value)) + 1;

    std::vector<std::size_t> count(static_cast<std::size_t>(range));
    for (int value : a) {
        ++count[static_cast<std::size_t>(
            static_cast<std::int64_t>(value) - static_cast<std::int64_t>(min_value))];
    }

    std::size_t out = 0;
    for (std::size_t offset = 0; offset < count.size(); ++offset) {
        const int value = static_cast<int>(static_cast<std::int64_t>(min_value) +
                                           static_cast<std::int64_t>(offset));
        for (std::size_t copies = count[offset]; copies > 0; --copies) {
            a[out++] = value;
        }
    }
}

static void counting_sort_compact_range(std::vector<int>& a) {
    if (a.empty()) return;

    const auto [min_it, max_it] = std::minmax_element(a.begin(), a.end());
    counting_sort_known_range(a, *min_it, *max_it);
}

static void adaptive_int_sort_sampled(std::vector<int>& a, std::size_t sample_limit) {
    if (a.size() < 2) return;

    const std::size_t sample_size = std::min(sample_limit, a.size());
    bool sample_ascending = true;
    bool sample_descending = true;
    std::size_t sample_descents = 0;
    int sample_min = a[0];
    int sample_max = a[0];

    for (std::size_t i = 1; i < sample_size; ++i) {
        sample_descents += a[i - 1] > a[i] ? 1 : 0;
        sample_ascending = sample_ascending && a[i - 1] <= a[i];
        sample_descending = sample_descending && a[i - 1] >= a[i];
        sample_min = std::min(sample_min, a[i]);
        sample_max = std::max(sample_max, a[i]);
    }

    const std::uint64_t sample_range = static_cast<std::uint64_t>(
        static_cast<std::int64_t>(sample_max) - static_cast<std::int64_t>(sample_min)) + 1;
    if (!sample_ascending && !sample_descending &&
        sample_descents > sample_size / 8 && sample_range > sample_size * 8ull) {
        radix_sort_i32_16bit_fused_counts(a);
        return;
    }

    bool ascending = true;
    bool descending = true;
    int min_value = a[0];
    int max_value = a[0];

    for (std::size_t i = 1; i < a.size(); ++i) {
        ascending = ascending && a[i - 1] <= a[i];
        descending = descending && a[i - 1] >= a[i];
        min_value = std::min(min_value, a[i]);
        max_value = std::max(max_value, a[i]);
    }

    if (ascending) return;
    if (descending) {
        std::reverse(a.begin(), a.end());
        return;
    }

    const std::uint64_t range = static_cast<std::uint64_t>(
        static_cast<std::int64_t>(max_value) - static_cast<std::int64_t>(min_value)) + 1;
    const std::uint64_t max_reasonable_range = std::max<std::uint64_t>(a.size() * 2ull, 1024ull);

    if (range <= max_reasonable_range && range <= 16ull * 1024ull * 1024ull) {
        counting_sort_known_range(a, min_value, max_value);
        return;
    }

    radix_sort_i32_16bit_fused_counts(a);
}

static void adaptive_int_sort_sampled_reuse(std::vector<int>& a, std::size_t sample_limit) {
    if (a.size() < 2) return;

    const std::size_t sample_size = std::min(sample_limit, a.size());
    bool sample_ascending = true;
    bool sample_descending = true;
    std::size_t sample_descents = 0;
    int sample_min = a[0];
    int sample_max = a[0];

    for (std::size_t i = 1; i < sample_size; ++i) {
        sample_descents += a[i - 1] > a[i] ? 1 : 0;
        sample_ascending = sample_ascending && a[i - 1] <= a[i];
        sample_descending = sample_descending && a[i - 1] >= a[i];
        sample_min = std::min(sample_min, a[i]);
        sample_max = std::max(sample_max, a[i]);
    }

    const std::uint64_t sample_range = static_cast<std::uint64_t>(
        static_cast<std::int64_t>(sample_max) - static_cast<std::int64_t>(sample_min)) + 1;
    if (!sample_ascending && !sample_descending &&
        sample_descents > sample_size / 8 && sample_range > sample_size * 8ull) {
        radix_sort_i32_16bit_fused_reuse(a);
        return;
    }

    bool ascending = true;
    bool descending = true;
    int min_value = a[0];
    int max_value = a[0];

    for (std::size_t i = 1; i < a.size(); ++i) {
        ascending = ascending && a[i - 1] <= a[i];
        descending = descending && a[i - 1] >= a[i];
        min_value = std::min(min_value, a[i]);
        max_value = std::max(max_value, a[i]);
    }

    if (ascending) return;
    if (descending) {
        std::reverse(a.begin(), a.end());
        return;
    }

    const std::uint64_t range = static_cast<std::uint64_t>(
        static_cast<std::int64_t>(max_value) - static_cast<std::int64_t>(min_value)) + 1;
    const std::uint64_t max_reasonable_range = std::max<std::uint64_t>(a.size() * 2ull, 1024ull);

    if (range <= max_reasonable_range && range <= 16ull * 1024ull * 1024ull) {
        counting_sort_known_range(a, min_value, max_value);
        return;
    }

    radix_sort_i32_16bit_fused_reuse(a);
}

static void adaptive_int_sort(std::vector<int>& a) {
    adaptive_int_sort_sampled(a, 16 * 1024);
}

static void adaptive_int_sort_16k(std::vector<int>& a) {
    adaptive_int_sort_sampled(a, 16 * 1024);
}

static void adaptive_int_sort_64k(std::vector<int>& a) {
    adaptive_int_sort_sampled(a, 64 * 1024);
}

static void adaptive_int_sort_reuse(std::vector<int>& a) {
    adaptive_int_sort_sampled_reuse(a, 16 * 1024);
}

static void adaptive_int_sort_parallel(std::vector<int>& a) {
    if (a.size() < 2) return;

    constexpr std::size_t sample_limit = 16 * 1024;
    const std::size_t sample_size = std::min(sample_limit, a.size());
    bool sample_ascending = true;
    bool sample_descending = true;
    std::size_t sample_descents = 0;
    int sample_min = a[0];
    int sample_max = a[0];

    for (std::size_t i = 1; i < sample_size; ++i) {
        sample_descents += a[i - 1] > a[i] ? 1 : 0;
        sample_ascending = sample_ascending && a[i - 1] <= a[i];
        sample_descending = sample_descending && a[i - 1] >= a[i];
        sample_min = std::min(sample_min, a[i]);
        sample_max = std::max(sample_max, a[i]);
    }

    const std::uint64_t sample_range = static_cast<std::uint64_t>(
        static_cast<std::int64_t>(sample_max) - static_cast<std::int64_t>(sample_min)) + 1;
    if (!sample_ascending && !sample_descending &&
        sample_descents > sample_size / 8 && sample_range > sample_size * 8ull) {
        radix_sort_i32_16bit_parallel(a);
        return;
    }

    adaptive_int_sort_sampled_reuse(a, sample_limit);
}

static std::vector<int> random_data(std::size_t n, std::uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(0, static_cast<int>(n * 4));
    std::vector<int> data(n);
    for (int& value : data) value = dist(rng);
    return data;
}

static std::vector<int> nearly_sorted_data(std::size_t n, std::uint32_t seed) {
    std::vector<int> data(n);
    std::iota(data.begin(), data.end(), 0);
    std::mt19937 rng(seed);
    std::uniform_int_distribution<std::size_t> pos(0, n - 1);
    const std::size_t swaps = std::max<std::size_t>(1, n / 100);
    for (std::size_t i = 0; i < swaps; ++i) {
        std::swap(data[pos(rng)], data[pos(rng)]);
    }
    return data;
}

static std::vector<int> sorted_data(std::size_t n) {
    std::vector<int> data(n);
    std::iota(data.begin(), data.end(), 0);
    return data;
}

static std::vector<int> reverse_data(std::size_t n) {
    std::vector<int> data(n);
    std::iota(data.rbegin(), data.rend(), 0);
    return data;
}

static std::vector<int> few_unique_data(std::size_t n, std::uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(0, 15);
    std::vector<int> data(n);
    for (int& value : data) value = dist(rng);
    return data;
}

static std::vector<std::int64_t> random_i64_data(std::size_t n, std::uint32_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<std::int64_t> dist(
        std::numeric_limits<std::int64_t>::min(),
        std::numeric_limits<std::int64_t>::max());
    std::vector<std::int64_t> data(n);
    for (std::int64_t& value : data) value = dist(rng);
    return data;
}

static std::vector<std::int64_t> sorted_i64_data(std::size_t n) {
    std::vector<std::int64_t> data(n);
    for (std::size_t i = 0; i < n; ++i) data[i] = static_cast<std::int64_t>(i);
    return data;
}

static std::vector<std::int64_t> reverse_i64_data(std::size_t n) {
    std::vector<std::int64_t> data(n);
    for (std::size_t i = 0; i < n; ++i) data[i] = static_cast<std::int64_t>(n - i);
    return data;
}

static std::vector<std::int64_t> few_unique_i64_data(std::size_t n, std::uint32_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<std::int64_t> dist(-8, 8);
    std::vector<std::int64_t> data(n);
    for (std::int64_t& value : data) value = dist(rng);
    return data;
}

static std::vector<Record> record_data(std::size_t n, std::uint32_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<std::int64_t> key_dist(-1000000000000ll, 1000000000000ll);
    std::uniform_int_distribution<std::uint32_t> category_dist(0, 255);
    std::uniform_real_distribution<double> score_dist(0.0, 1.0);
    std::vector<Record> data(n);

    for (std::size_t i = 0; i < n; ++i) {
        Record item{};
        item.key = key_dist(rng);
        item.category = category_dist(rng);
        item.score = score_dist(rng);
        const std::string label = "item-" + std::to_string(rng());
        std::copy_n(label.begin(), std::min(label.size(), item.label.size() - 1), item.label.begin());
        data[i] = item;
    }

    return data;
}

static std::vector<std::string> string_data(std::size_t n, std::uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> len_dist(8, 28);
    std::uniform_int_distribution<int> char_dist(0, 25);
    std::vector<std::string> data;
    data.reserve(n);

    for (std::size_t i = 0; i < n; ++i) {
        std::string value;
        const int len = len_dist(rng);
        value.reserve(static_cast<std::size_t>(len));
        for (int j = 0; j < len; ++j) {
            value.push_back(static_cast<char>('a' + char_dist(rng)));
        }
        data.push_back(std::move(value));
    }

    return data;
}

static double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

static double percentile(std::vector<double> values, double pct) {
    std::sort(values.begin(), values.end());
    const std::size_t index = std::min(values.size() - 1,
        static_cast<std::size_t>((pct / 100.0) * static_cast<double>(values.size() - 1)));
    return values[index];
}

static MemoryStatus memory_status() {
    std::ifstream status("/proc/self/status");
    MemoryStatus memory;
    std::string line;

    while (std::getline(status, line)) {
        std::istringstream fields(line);
        std::string key;
        std::size_t value = 0;
        std::string unit;
        fields >> key >> value >> unit;
        if (key == "VmRSS:") memory.rss_kb = value;
        if (key == "VmHWM:") memory.hwm_kb = value;
    }

    return memory;
}

static std::vector<std::pair<std::string, std::function<void(std::vector<int>&)>>> algorithms() {
    return {
        {"plain_merge", plain_merge_sort},
        {"tuned_merge", tuned_merge_sort},
        {"bottom_up_merge", bottom_up_merge_sort},
        {"radix_int_sort", radix_sort_i32},
        {"radix16_int_sort", radix_sort_i32_16bit},
        {"radix16_fused", radix_sort_i32_16bit_fused_counts},
        {"radix16_reuse", radix_sort_i32_16bit_fused_reuse},
        {"radix16_parallel", radix_sort_i32_16bit_parallel},
        {"radix11_fused", radix_sort_i32_11bit_fused_counts},
        {"radix16_openmp", radix_sort_i32_16bit_openmp},
        {"radix16_pool", radix_sort_i32_16bit_pool},
        {"radix16_pool2", radix_sort_i32_16bit_pool2},
        {"radix16_pool4", radix_sort_i32_16bit_pool4},
        {"counting_sort", counting_sort_compact_range},
        {"adaptive_int_sort", adaptive_int_sort},
        {"adaptive_16k", adaptive_int_sort_16k},
        {"adaptive_64k", adaptive_int_sort_64k},
        {"adaptive_reuse", adaptive_int_sort_reuse},
        {"adaptive_parallel", adaptive_int_sort_parallel},
        {"std_stable_sort", [](std::vector<int>& a) { std::stable_sort(a.begin(), a.end()); }},
        {"std_sort", [](std::vector<int>& a) { std::sort(a.begin(), a.end()); }}
    };
}

static std::vector<int> make_dataset(std::string_view dataset, std::size_t n, std::uint32_t seed) {
    if (dataset == "random") return random_data(n, seed);
    if (dataset == "sorted") return sorted_data(n);
    if (dataset == "nearly_sorted") return nearly_sorted_data(n, seed);
    if (dataset == "reverse") return reverse_data(n);
    if (dataset == "few_unique") return few_unique_data(n, seed);
    throw std::invalid_argument("unknown dataset");
}

static Result benchmark(std::string_view name, std::string_view dataset,
                        const std::vector<int>& input,
                        const std::function<void(std::vector<int>&)>& sorter,
                        int rounds) {
    std::vector<double> times;
    times.reserve(static_cast<std::size_t>(rounds));
    bool ok = true;

    for (int round = 0; round < rounds; ++round) {
        std::vector<int> data = input;
        const auto start = Clock::now();
        sorter(data);
        const auto stop = Clock::now();
        ok = ok && std::is_sorted(data.begin(), data.end());
        times.push_back(std::chrono::duration<double, std::milli>(stop - start).count());
    }

    return {std::string(name), std::string(dataset), input.size(), median(times), ok};
}

template <typename T, typename Sorted>
static Result benchmark_typed(std::string_view name, std::string_view dataset,
                              const std::vector<T>& input,
                              const std::function<void(std::vector<T>&)>& sorter,
                              int rounds,
                              Sorted sorted) {
    std::vector<double> times;
    times.reserve(static_cast<std::size_t>(rounds));
    bool ok = true;

    for (int round = 0; round < rounds; ++round) {
        std::vector<T> data = input;
        const auto start = Clock::now();
        sorter(data);
        const auto stop = Clock::now();
        ok = ok && sorted(data);
        times.push_back(std::chrono::duration<double, std::milli>(stop - start).count());
    }

    return {std::string(name), std::string(dataset), input.size(), median(times), ok};
}

static void print_header(std::size_t n, int rounds, std::string_view mode) {
    std::cout << "mode=" << mode << " n=" << n << " rounds=" << rounds << "\n";
    std::cout << std::left
              << std::setw(16) << "dataset"
              << std::setw(18) << "algorithm"
              << std::right
              << std::setw(12) << "median_ms"
              << std::setw(10) << "sorted"
              << "\n";
}

static void print_result(const Result& result) {
    std::cout << std::left
              << std::setw(16) << result.dataset
              << std::setw(18) << result.name
              << std::right
              << std::setw(12) << std::fixed << std::setprecision(3) << result.median_ms
              << std::setw(10) << (result.sorted ? "yes" : "no")
              << "\n";
}

static int run_i64_benchmark(std::size_t n, int rounds) {
    constexpr std::uint32_t seed = 0xC0FFEE;
    const std::vector<std::pair<std::string, std::vector<std::int64_t>>> datasets = {
        {"random64", random_i64_data(n, seed)},
        {"sorted64", sorted_i64_data(n)},
        {"reverse64", reverse_i64_data(n)},
        {"few_unique64", few_unique_i64_data(n, seed)}
    };
    const std::vector<std::pair<std::string, std::function<void(std::vector<std::int64_t>&)>>> algorithm_list = {
        {"radix64_16bit", radix_sort_i64_16bit},
        {"adaptive_i64", adaptive_i64_sort},
        {"std_stable_sort", [](std::vector<std::int64_t>& a) { std::stable_sort(a.begin(), a.end()); }},
        {"std_sort", [](std::vector<std::int64_t>& a) { std::sort(a.begin(), a.end()); }}
    };
    const auto sorted = [](const std::vector<std::int64_t>& data) {
        return std::is_sorted(data.begin(), data.end());
    };

    print_header(n, rounds, "i64");
    for (const auto& [dataset_name, input] : datasets) {
        for (const auto& [algorithm_name, sorter] : algorithm_list) {
            print_result(benchmark_typed(algorithm_name, dataset_name, input, sorter, rounds, sorted));
        }
        std::cout << "\n";
    }

    return 0;
}

static int run_object_benchmark(std::size_t n, int rounds) {
    constexpr std::uint32_t seed = 0xC0FFEE;
    const std::vector<std::pair<std::string, std::vector<Record>>> record_datasets = {
        {"records_key", record_data(n, seed)}
    };
    const std::vector<std::pair<std::string, std::function<void(std::vector<Record>&)>>> record_algorithms = {
        {"radix_key64", radix_sort_records_by_key},
        {"std_sort_key", [](std::vector<Record>& a) {
             std::sort(a.begin(), a.end(), [](const Record& lhs, const Record& rhs) {
                 return lhs.key < rhs.key;
             });
         }},
        {"std_stable_key", [](std::vector<Record>& a) {
             std::stable_sort(a.begin(), a.end(), [](const Record& lhs, const Record& rhs) {
                 return lhs.key < rhs.key;
             });
         }},
        {"std_sort_multi", [](std::vector<Record>& a) {
             std::sort(a.begin(), a.end(), [](const Record& lhs, const Record& rhs) {
                 if (lhs.category != rhs.category) return lhs.category < rhs.category;
                 if (lhs.score != rhs.score) return lhs.score < rhs.score;
                 return lhs.key < rhs.key;
             });
         }}
    };
    const auto records_sorted = [](const std::vector<Record>& data) {
        return std::is_sorted(data.begin(), data.end(), [](const Record& lhs, const Record& rhs) {
            return lhs.key < rhs.key;
        });
    };
    const auto records_multi_sorted = [](const std::vector<Record>& data) {
        return std::is_sorted(data.begin(), data.end(), [](const Record& lhs, const Record& rhs) {
            if (lhs.category != rhs.category) return lhs.category < rhs.category;
            if (lhs.score != rhs.score) return lhs.score < rhs.score;
            return lhs.key < rhs.key;
        });
    };

    print_header(n, rounds, "objects");
    for (const auto& [dataset_name, input] : record_datasets) {
        for (const auto& [algorithm_name, sorter] : record_algorithms) {
            if (algorithm_name == "std_sort_multi") {
                print_result(benchmark_typed(algorithm_name, dataset_name, input, sorter, rounds,
                                             records_multi_sorted));
            } else {
                print_result(benchmark_typed(algorithm_name, dataset_name, input, sorter, rounds,
                                             records_sorted));
            }
        }
        std::cout << "\n";
    }

    const std::vector<std::pair<std::string, std::vector<std::string>>> string_datasets = {
        {"strings", string_data(n, seed)}
    };
    const std::vector<std::pair<std::string, std::function<void(std::vector<std::string>&)>>> string_algorithms = {
        {"std_sort", [](std::vector<std::string>& a) { std::sort(a.begin(), a.end()); }},
        {"std_stable_sort", [](std::vector<std::string>& a) { std::stable_sort(a.begin(), a.end()); }}
    };
    const auto strings_sorted = [](const std::vector<std::string>& data) {
        return std::is_sorted(data.begin(), data.end());
    };

    for (const auto& [dataset_name, input] : string_datasets) {
        for (const auto& [algorithm_name, sorter] : string_algorithms) {
            print_result(benchmark_typed(algorithm_name, dataset_name, input, sorter, rounds, strings_sorted));
        }
        std::cout << "\n";
    }

    return 0;
}

static int run_stress(int argc, char** argv) {
    constexpr std::uint32_t seed = 0xC0FFEE;
    const std::string algorithm_name = argc > 2 ? argv[2] : "adaptive_int_sort";
    const std::string dataset_name = argc > 3 ? argv[3] : "random";
    const std::size_t n = argc > 4 ? static_cast<std::size_t>(std::stoull(argv[4])) : 1000000;
    const double seconds = argc > 5 ? std::stod(argv[5]) : 60.0;
    const std::size_t pool_size = argc > 6 ? static_cast<std::size_t>(std::stoull(argv[6])) : 8;
    const std::size_t verify_every = argc > 7 ? static_cast<std::size_t>(std::stoull(argv[7])) : 1;
    const std::size_t memory_every = argc > 8 ? static_cast<std::size_t>(std::stoull(argv[8])) : 1;

    auto algorithm_list = algorithms();
    auto it = std::find_if(algorithm_list.begin(), algorithm_list.end(),
        [&](const auto& item) { return item.first == algorithm_name; });
    if (it == algorithm_list.end()) {
        std::cerr << "Unknown algorithm: " << algorithm_name << "\n";
        return 2;
    }

    std::vector<std::vector<int>> pool;
    pool.reserve(pool_size);
    for (std::size_t i = 0; i < pool_size; ++i) {
        pool.push_back(make_dataset(dataset_name, n, seed + static_cast<std::uint32_t>(i * 7919u)));
    }

    std::vector<int> work;
    work.reserve(n);
    std::vector<double> latencies_ms;
    latencies_ms.reserve(8192);
    std::size_t operations = 0;
    std::size_t failures = 0;
    MemoryStatus peak = memory_status();

    const auto start = Clock::now();
    auto now = start;
    while (std::chrono::duration<double>(now - start).count() < seconds) {
        const std::vector<int>& input = pool[operations % pool.size()];

        const auto op_start = Clock::now();
        work = input;
        it->second(work);
        const auto op_stop = Clock::now();

        if (verify_every > 0 && operations % verify_every == 0) {
            failures += std::is_sorted(work.begin(), work.end()) ? 0 : 1;
        }
        latencies_ms.push_back(std::chrono::duration<double, std::milli>(op_stop - op_start).count());
        ++operations;

        if (memory_every > 0 && operations % memory_every == 0) {
            const MemoryStatus current = memory_status();
            peak.rss_kb = std::max(peak.rss_kb, current.rss_kb);
            peak.hwm_kb = std::max(peak.hwm_kb, current.hwm_kb);
        }
        now = Clock::now();
    }

    const double elapsed = std::chrono::duration<double>(Clock::now() - start).count();
    const double sorts_per_sec = static_cast<double>(operations) / elapsed;
    const double million_ints_per_sec = (static_cast<double>(operations) * static_cast<double>(n)) /
                                        elapsed / 1000000.0;
    const double input_mb = static_cast<double>(n * sizeof(int)) / (1024.0 * 1024.0);
    const double pool_mb = input_mb * static_cast<double>(pool.size());
    const double median_ms = median(latencies_ms);
    const double p95_ms = percentile(latencies_ms, 95.0);

    std::cout << "stress_seconds=" << std::fixed << std::setprecision(3) << elapsed
              << " algorithm=" << algorithm_name
              << " dataset=" << dataset_name
              << " n=" << n
              << " pool=" << pool.size()
              << " verify_every=" << verify_every
              << " memory_every=" << memory_every
              << "\n";
    std::cout << "operations=" << operations
              << " failures=" << failures
              << " sorts_per_sec=" << std::setprecision(3) << sorts_per_sec
              << " million_ints_per_sec=" << million_ints_per_sec
              << "\n";
    std::cout << "median_ms=" << median_ms
              << " p95_ms=" << p95_ms
              << " input_mb=" << input_mb
              << " pool_input_mb=" << pool_mb
              << "\n";
    std::cout << "rss_peak_mb=" << static_cast<double>(peak.rss_kb) / 1024.0
              << " hwm_mb=" << static_cast<double>(peak.hwm_kb) / 1024.0
              << "\n";

    return failures == 0 ? 0 : 1;
}

int main(int argc, char** argv) {
    if (argc > 1 && std::string_view(argv[1]) == "stress") {
        return run_stress(argc, argv);
    }

    if (argc > 1 && std::string_view(argv[1]) == "i64") {
        const std::size_t n = argc > 2 ? static_cast<std::size_t>(std::stoull(argv[2])) : 200000;
        const int rounds = argc > 3 ? std::stoi(argv[3]) : 7;
        return run_i64_benchmark(n, rounds);
    }

    if (argc > 1 && std::string_view(argv[1]) == "objects") {
        const std::size_t n = argc > 2 ? static_cast<std::size_t>(std::stoull(argv[2])) : 100000;
        const int rounds = argc > 3 ? std::stoi(argv[3]) : 5;
        return run_object_benchmark(n, rounds);
    }

    if (argc > 1 && std::string_view(argv[1]) == "all") {
        const std::size_t n = argc > 2 ? static_cast<std::size_t>(std::stoull(argv[2])) : 100000;
        const int rounds = argc > 3 ? std::stoi(argv[3]) : 5;
        int status = 0;
        status |= run_i64_benchmark(n, rounds);
        status |= run_object_benchmark(n, rounds);
        return status;
    }

    const std::size_t n = argc > 1 ? static_cast<std::size_t>(std::stoull(argv[1])) : 200000;
    const int rounds = argc > 2 ? std::stoi(argv[2]) : 7;
    constexpr std::uint32_t seed = 0xC0FFEE;

    const std::vector<std::pair<std::string, std::vector<int>>> datasets = {
        {"random", random_data(n, seed)},
        {"sorted", sorted_data(n)},
        {"nearly_sorted", nearly_sorted_data(n, seed)},
        {"reverse", reverse_data(n)},
        {"few_unique", few_unique_data(n, seed)}
    };

    const auto algorithm_list = algorithms();

    print_header(n, rounds, "i32");

    for (const auto& [dataset_name, input] : datasets) {
        for (const auto& [algorithm_name, sorter] : algorithm_list) {
            print_result(benchmark(algorithm_name, dataset_name, input, sorter, rounds));
        }
        std::cout << "\n";
    }
}
