#include <iostream>
#include <stdexcept>
#include "../core/tokenizer.h"

int main() {
    try {
        const std::string model_dir = "/root/gpt2-tamil-124m";
        const std::string text = "\xE0\xAE\xAF\xE0\xAE\xBE\xE0\xAE\xA4\xE0\xAF\x81\xE0\xAE\xAE\xE0\xAF\x8D\x20\xE0\xAE\x8A\xE0\xAE\xB0\xE0\xAF\x87\x20\xE0\xAE\xAF\xE0\xAE\xBE\xE0\xAE\xB5\xE0\xAE\xB0\xE0\xAF\x81\xE0\xAE\xAE\xE0\xAF\x8D\x20\xE0\xAE\x95\xE0\xAF\x87\xE0\xAE\xB3\xE0\xAE\xBF\xE0\xAE\xB0\xE0\xAF\x8D";

        ops::TokenizerLoadOptions opts;
        opts.model_type = "gpt2";

        auto tok = ops::TokenizerFactory::from_pretrained(model_dir, opts);

        if (!tok) {
            throw std::runtime_error("Tokenizer creation failed");
        }

        auto ids = tok->encode(text);

        std::cout << "Native tokenizer vocab: "
                  << tok->get_vocab_size() << "\n";

        std::cout << "Token count: " << ids.size() << "\n";
        std::cout << "IDs: [";

        for (size_t i = 0; i < ids.size(); ++i) {
            if (i) std::cout << ", ";
            std::cout << ids[i];
        }

        std::cout << "]\n";
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "[ERROR] " << e.what() << "\n";
        return 1;
    }
}
