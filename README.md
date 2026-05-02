# Algorithms

Personal algorithm experiments by Jose Diaz.

This repo starts with a focused sorting experiment: when data has known fixed-width keys, can a specialized sorter beat a general-purpose comparison sort?

## What Is Here

- `radix16.cpp` - compact standalone 16-bit radix sort example for signed 32-bit integers.
- `bench_sort.cpp` - C++ benchmark and stress-test harness for merge sort, `std::sort`, radix variants, counting sort, adaptive integer sorting, 64-bit integers, records, and strings.
- `index.html` - slideshow-style explanation with a live JavaScript typed-array benchmark.

## Core Idea

General-purpose sorting usually compares values:

```cpp
std::sort(values.begin(), values.end());
```

That is flexible, but for plain fixed-width integer data we can use the known key width. The 32-bit radix16 sorter splits each integer into two 16-bit chunks:

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

Test machine:

- CPU: 12th Gen Intel Core i7-12700K, 12 cores / 20 threads, up to 5.0 GHz
- Memory: 31 GiB RAM
- OS: Linux Mint 22.1 `xia`, kernel `6.8.0-110-generic`
- Compiler: `g++ 13.3.0`
- GPU present but not used by this CPU benchmark: NVIDIA GeForce RTX 4080 SUPER, 16 GiB VRAM

The 60-second utility stress test on 1,000,000 random integers per batch measured:

```text
adaptive_int_sort: 169.153 million ints/sec
std_sort:           20.780 million ints/sec
```

That is about 8.1x faster for this workload.

A short sustained run of the reusable single-thread radix path reached about 178 million ints/sec.

## 64-Bit And Object Sorting

The benchmark also has extended modes:

```bash
./bench_sort i64 200000 3
./bench_sort objects 100000 3
```

The 64-bit integer path uses four 16-bit radix passes. On full-width random `int64_t` values, `std::sort` can still win because the radix path does twice as many passes as the 32-bit sorter and moves more memory. On already sorted or reverse-sorted 64-bit data, the adaptive path wins by detecting the shape first. On low-cardinality 64-bit data, radix-style sorting is competitive again.

For records, the benchmark includes both normal comparator sorting and a radix path that sorts whole records by an extracted signed 64-bit key. That is the important distinction for real programs: general object sorting still needs `std::sort` or `std::stable_sort`, but objects with a simple numeric key can sometimes use a specialized key sorter.

For strings and multi-field object comparators, this repo keeps using general comparison sorting. A radix sorter is not automatically the best answer once the ordering rule depends on variable-length text, multiple fields, locale rules, custom comparators, or stability.

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

Run the extended benchmark modes:

```bash
./bench_sort i64 200000 3
./bench_sort objects 100000 3
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

This is not a universal replacement for standard sorting. It is best for known fixed-width numeric keys. For strings, custom comparators, stable ordering requirements, or unknown data shapes, standard library sorting remains the right default. For objects, the best approach depends on whether the sort order can be reduced to a simple key extraction.

I plan to keep experimenting with more algorithms, data shapes, and memory strategies.
