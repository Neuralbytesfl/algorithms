# Algorithms

Personal algorithm experiments by Jose Diaz.

This repo starts with a focused sorting experiment: if the data is known to be 32-bit signed integers, can a specialized integer sorter beat a general-purpose comparison sort?

## What Is Here

- `radix16.cpp` - compact standalone 16-bit radix sort example for signed 32-bit integers.
- `bench_sort.cpp` - C++ benchmark and stress-test harness for merge sort, `std::sort`, radix variants, counting sort, and adaptive integer sorting.
- `index.html` - slideshow-style explanation with a live JavaScript typed-array benchmark.

## Core Idea

General-purpose sorting usually compares values:

```cpp
std::sort(values.begin(), values.end());
```

That is flexible, but for plain `int32_t` data we can use the fixed 32-bit key width. The radix16 sorter splits each integer into two 16-bit chunks:

1. Count buckets for the low 16 bits.
2. Scatter values into scratch memory.
3. Count buckets for the high 16 bits.
4. Scatter values back into final sorted order.

For signed integers, the key is transformed with:

```cpp
uint32_t key = uint32_t(value) ^ 0x80000000u;
```

That maps signed integer ordering into unsigned key ordering, so negative and positive values sort correctly.

## Why It Can Be Faster

The specialized path avoids repeated comparison branching. It mostly performs linear scans, bucket counts, prefix sums, and contiguous memory writes.

The tradeoff is memory: radix/counting-style sorting uses scratch buffers and bucket counters. In the stress test, the adaptive integer path used about 43 MB peak RSS for 1,000,000-int batches.

## Measured Results

On this machine, with:

```bash
g++ -O3 -march=native -std=c++20 -pthread bench_sort.cpp -o bench_sort
```

The 60-second utility stress test on 1,000,000 random integers per batch measured:

```text
adaptive_int_sort: 169.153 million ints/sec
std_sort:           20.780 million ints/sec
```

That is about 8.1x faster for this workload.

A short sustained run of the reusable single-thread radix path reached about 178 million ints/sec.

## Run The Compact Example

```bash
g++ -O3 -march=native -std=c++20 radix16.cpp -o radix16
./radix16
```

Expected output:

```text
-9 -2 0 1 4 7
```

## Run The Benchmark

```bash
g++ -O3 -march=native -std=c++20 -pthread bench_sort.cpp -o bench_sort
./bench_sort 1000000 5
```

Run a sustained stress test:

```bash
./bench_sort stress adaptive_int_sort random 1000000 60 8
./bench_sort stress std_sort random 1000000 60 8
```

## Open The Slideshow

Open `index.html` in a browser. It includes:

- the project story
- algorithm explanation
- measured C++ results
- a live JavaScript typed-array benchmark

## Scope

This is not a universal replacement for standard sorting. It is designed for known 32-bit integer data. For objects, strings, custom comparators, stable ordering requirements, or unknown data shapes, standard library sorting remains the right default.

I plan to keep experimenting with more algorithms, data shapes, and memory strategies.
