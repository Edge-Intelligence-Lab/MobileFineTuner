#include <cstdint>
#include <iomanip>
#include <iostream>
#include <vector>

#include "../core/tensor.h"
#include "../core/dtype.h"

using namespace ops;

int main() {
    try {
        auto t = std::make_shared<Tensor>(
            std::vector<int64_t>{4}, kBFloat16, kCPU);

        uint16_t* p = t->data<uint16_t>();

        const float values[4] = {0.9f, 1.9f, 2.9f, 3.9f};

        for (int i = 0; i < 4; ++i) {
            p[i] = float32_to_bf16_bits(values[i]);
        }

        std::cout << "dtype=" << DTypeUtils::to_string(t->dtype()) << "\n";

        bool ok = true;

        for (int i = 0; i < 4; ++i) {
            float decoded = bf16_bits_to_float32(p[i]);

            std::cout << "[" << i << "] bits=0x"
                      << std::hex << std::setw(4) << std::setfill('0') << p[i]
                      << std::dec
                      << " value=" << std::setprecision(9) << decoded
                      << "\n";

            if (decoded != bf16_bits_to_float32(
                                float32_to_bf16_bits(values[i]))) {
                ok = false;
            }
        }

        if (!ok) {
            std::cerr << "[FAIL] Direct BF16 storage test failed.\n";
            return 1;
        }

        std::cout << "[PASS] Direct BF16 storage/write/read works.\n";
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "[ERROR] " << e.what() << "\n";
        return 2;
    }
}
