#include <algorithm>
#include <cstdint>
#include <iostream>
#include <vector>

void radix16(std::vector<int32_t>& values) {
    constexpr size_t radix = 1u << 16u;
    std::vector<int32_t> scratch(values.size());
    std::vector<size_t> count(radix);

    for (unsigned shift : {0u, 16u}) {
        std::fill(count.begin(), count.end(), 0);

        for (int32_t value : values) {
            const uint32_t key = static_cast<uint32_t>(value) ^ 0x80000000u;
            ++count[(key >> shift) & 0xffffu];
        }

        for (size_t i = 1; i < radix; ++i) {
            count[i] += count[i - 1];
        }

        for (size_t i = values.size(); i-- > 0;) {
            const int32_t value = values[i];
            const uint32_t key = static_cast<uint32_t>(value) ^ 0x80000000u;
            scratch[--count[(key >> shift) & 0xffffu]] = value;
        }

        values.swap(scratch);
    }
}

int main() {
    std::vector<int32_t> values = {7, -2, 4, 1, -9, 0};
    radix16(values);

    for (int32_t value : values) {
        std::cout << value << ' ';
    }
    std::cout << '\n';
}
