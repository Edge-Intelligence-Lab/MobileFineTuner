/**
 * @file wikitext2_dataset.cpp
 * @brief WikiText-2 dataset implementation
 */

#include "wikitext2_dataset.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <filesystem>
#include <cctype>
#include <iterator>

namespace ops {

namespace {

bool extract_int64_field(const std::string& text,
                         const std::string& key,
                         int64_t& out,
                         size_t search_from = 0) {
    auto key_pos = text.find("\"" + key + "\"", search_from);
    if (key_pos == std::string::npos) return false;
    auto colon = text.find(':', key_pos);
    if (colon == std::string::npos) return false;
    size_t pos = colon + 1;
    while (pos < text.size() && std::isspace(static_cast<unsigned char>(text[pos]))) ++pos;
    size_t end = pos;
    if (end < text.size() && text[end] == '-') ++end;
    while (end < text.size() && std::isdigit(static_cast<unsigned char>(text[end]))) ++end;
    if (end <= pos) return false;
    try {
        out = std::stoll(text.substr(pos, end - pos));
        return true;
    } catch (...) {
        return false;
    }
}

bool extract_bool_field(const std::string& text,
                        const std::string& key,
                        bool& out,
                        size_t search_from = 0) {
    auto key_pos = text.find("\"" + key + "\"", search_from);
    if (key_pos == std::string::npos) return false;
    auto colon = text.find(':', key_pos);
    if (colon == std::string::npos) return false;
    size_t pos = colon + 1;
    while (pos < text.size() && std::isspace(static_cast<unsigned char>(text[pos]))) ++pos;
    if (text.compare(pos, 4, "true") == 0) {
        out = true;
        return true;
    }
    if (text.compare(pos, 5, "false") == 0) {
        out = false;
        return true;
    }
    return false;
}

bool extract_split_range(const std::string& text,
                         const std::string& name,
                         size_t& offset,
                         size_t& length) {
    auto split_pos = text.find("\"" + name + "\"");
    if (split_pos == std::string::npos) return false;
    int64_t off = 0;
    int64_t len = 0;
    if (!extract_int64_field(text, "offset", off, split_pos)) return false;
    if (!extract_int64_field(text, "length", len, split_pos)) return false;
    offset = static_cast<size_t>(std::max<int64_t>(0, off));
    length = static_cast<size_t>(std::max<int64_t>(0, len));
    return true;
}

std::string resolve_meta_path(const std::string& bin_path,
                              const std::string& provided_meta) {
    namespace fs = std::filesystem;
    if (!provided_meta.empty()) {
        return provided_meta;
    }
    fs::path bin(bin_path);
    fs::path dir = bin.parent_path();
    std::vector<std::string> candidates;
    candidates.push_back((dir / "meta.json").string());
    candidates.push_back(bin_path + ".json");
    candidates.push_back(bin_path + ".meta.json");
    if (bin.has_extension()) {
        fs::path replaced = bin;
        replaced.replace_extension(".json");
        candidates.push_back(replaced.string());
    }
    for (const auto& cand : candidates) {
        std::ifstream f(cand);
        if (f.good()) return cand;
    }
    return "";
}

}  // namespace

static std::string trim(const std::string& s) {
    size_t l = 0, r = s.size();
    while (l < r && std::isspace(static_cast<unsigned char>(s[l]))) ++l;
    while (r > l && std::isspace(static_cast<unsigned char>(s[r-1]))) --r;
    return s.substr(l, r-l);
}

WikiText2Dataset::WikiText2Dataset(const WT2Config& cfg, GPT2BPETokenizer* tok)
    : cfg_(cfg), tok_(tok), cursor_(0), rng_(cfg.seed) {
    encode_fn_ = [tok](const std::string& text) {
        auto enc = tok->encode(text, false, 0, false);
        return std::vector<int32_t>(enc.begin(), enc.end());
    };
}

WikiText2Dataset::WikiText2Dataset(const WT2Config& cfg,
                                   std::function<std::vector<int32_t>(const std::string&)> encode_fn)
    : cfg_(cfg), tok_(nullptr), encode_fn_(std::move(encode_fn)),
      cursor_(0), rng_(cfg.seed) {
}

std::vector<std::string> WikiText2Dataset::read_lines_for_split(Split split) const {
    std::string path;
    if (split == Split::Train) path = cfg_.train_path;
    else if (split == Split::Valid) path = cfg_.valid_path;
    else path = cfg_.test_path;

    std::ifstream in(path);
    if (!in) throw std::runtime_error("Open WT2 file failed: " + path);

    std::vector<std::string> lines;
    std::string line;
    while (std::getline(in, line)) {
        // Raw mode: allow empty lines; skip if insert_eos_between_lines is false
        if (!cfg_.insert_eos_between_lines && trim(line).empty()) continue;
        lines.emplace_back(line);
    }
    return lines;
}

std::vector<int32_t> WikiText2Dataset::tokenize_and_pack(const std::vector<std::string>& lines) const {
    std::vector<int32_t> ids;
    ids.reserve(lines.size() * 16); // rough estimate

    for (size_t i = 0; i < lines.size(); ++i) {
        // Use GPT2BPETokenizer (returns std::vector<int>)
        auto enc = encode_fn_(lines[i]);
        ids.insert(ids.end(), enc.begin(), enc.end());
        
        if (cfg_.insert_eos_between_lines) {
            ids.push_back(static_cast<int32_t>(cfg_.eos_id));
        }
    }
    
    // Ensure trailing token for label shift
    if (ids.empty() || ids.back() != cfg_.eos_id) {
        ids.push_back(cfg_.eos_id);
    }

    float frac = std::clamp(cfg_.data_fraction, 0.0f, 1.0f);
    if (frac < 1.0f) {
        size_t min_tokens = static_cast<size_t>(cfg_.seq_len + 1);
        size_t limit = static_cast<size_t>(static_cast<double>(ids.size()) * frac);
        limit = std::max(limit, min_tokens);
        limit = std::min(limit, ids.size());
        ids.resize(limit);
        if (ids.empty() || ids.back() != cfg_.eos_id) {
            ids.push_back(cfg_.eos_id);
        }
    }
    
    return ids;
}

void WikiText2Dataset::build_chunk_indices(const std::vector<int32_t>& ids) {
    starts_.clear();
    const int S = cfg_.seq_len;
    int stride = (cfg_.stride <= 0) ? S : cfg_.stride;

    // Need S+1 tokens to form [S] inputs + [S] labels (shifted)
    const long long need = static_cast<long long>(S) + 1;
    long long N = static_cast<long long>(ids.size());

    for (long long s = 0; s + need <= N; s += stride) {
        starts_.push_back(static_cast<size_t>(s));
    }

    if (!cfg_.drop_last) {
        // Add a tail chunk (possibly short); padding is handled in get_batch
        if (starts_.empty() || starts_.back() + need < static_cast<size_t>(N)) {
            long long s = std::max(0LL, N - need);
            if (starts_.empty() || starts_.back() != static_cast<size_t>(s)) {
                starts_.push_back(static_cast<size_t>(s));
            }
        }
    }

    // Build traversal order (can be shuffled)
    order_.resize(starts_.size());
    std::iota(order_.begin(), order_.end(), 0);
}

void WikiText2Dataset::load(Split split) {
    ids_.clear();
    starts_.clear();
    order_.clear();
    cursor_ = 0;
    ids_global_offset_ = 0;
    total_tokens_ = 0;
    pretokenized_mode_ = !cfg_.pretokenized_path.empty();
    jsonl_mode_ = (!cfg_.jsonl_train.empty() || !cfg_.jsonl_valid.empty() || !cfg_.jsonl_test.empty());

    if (jsonl_mode_) {
        load_jsonl_split(split);
        // Build traversal order
        order_.resize(jsonl_samples_.size());
        std::iota(order_.begin(), order_.end(), 0);
        if (split == Split::Train && cfg_.shuffle_train) {
            shuffle();
        }
        return;
    } else if (pretokenized_mode_) {
        cfg_.streaming_mode = false;  // Pretokenized mode keeps required slices resident
        ensure_pretokenized_meta_loaded();
        load_pretokenized_split(split);
        if (split == Split::Train && cfg_.shuffle_train) {
            shuffle();
        }
        return;
    }
    
    // Save file path
    if (split == Split::Train) current_file_path_ = cfg_.train_path;
    else if (split == Split::Valid) current_file_path_ = cfg_.valid_path;
    else current_file_path_ = cfg_.test_path;
    
    if (cfg_.streaming_mode) {
        // Streaming mode: pre-scan to compute true token offsets; do not hold full text
        precompute_chunk_offsets();
        
        // Build traversal order
        order_.resize(starts_.size());
        std::iota(order_.begin(), order_.end(), 0);
        
        if (split == Split::Train && cfg_.shuffle_train) {
            shuffle();
        }
        
        // Preload first window (cover the first few chunks)
        if (!starts_.empty()) {
            size_t first_window_end = std::min(
                starts_[0] + cfg_.max_cache_tokens, 
                total_tokens_
            );
            load_window_from_file(starts_[0], first_window_end - starts_[0]);
        }
    } else {
        // Non-streaming: load all at once
        auto lines = read_lines_for_split(split);
        ids_ = tokenize_and_pack(lines);
        build_chunk_indices(ids_);
        
        if (split == Split::Train) {
            shuffle();
        }
    }
}

size_t WikiText2Dataset::num_sequences() const {
    if (jsonl_mode_) {
        return jsonl_samples_.size();
    }
    return starts_.size();
}

void WikiText2Dataset::shuffle() {
    std::shuffle(order_.begin(), order_.end(), rng_);
    cursor_ = 0;
}

void WikiText2Dataset::reset_cursor() {
    cursor_ = 0;
}

bool WikiText2Dataset::parse_jsonl_ids_mask(const std::string& line,
                                            std::vector<int32_t>& ids_out,
                                            std::vector<uint8_t>& mask_out) {
    auto find_array = [](const std::string& s, const std::string& key, size_t from) -> std::pair<size_t,size_t> {
        size_t k = s.find("\"" + key + "\"", from);
        if (k == std::string::npos) return {std::string::npos, std::string::npos};
        size_t lb = s.find('[', k);
        size_t rb = s.find(']', lb);
        if (lb == std::string::npos || rb == std::string::npos) return {std::string::npos, std::string::npos};
        return {lb, rb};
    };
    auto parse_ints = [](const std::string& s, size_t lb, size_t rb) -> std::vector<long long> {
        std::vector<long long> vals;
        std::string cur;
        for (size_t i = lb + 1; i < rb; ++i) {
            char c = s[i];
            if (c == ',' || c == ' ' || c == '\t') {
                if (!cur.empty()) {
                    try { vals.push_back(std::stoll(cur)); } catch (...) {}
                    cur.clear();
                }
            } else {
                cur.push_back(c);
            }
        }
        if (!cur.empty()) {
            try { vals.push_back(std::stoll(cur)); } catch (...) {}
        }
        return vals;
    };

    auto ids_pos = find_array(line, "ids", 0);
    if (ids_pos.first == std::string::npos) return false;
    auto mask_pos = find_array(line, "mask", ids_pos.second);
    if (mask_pos.first == std::string::npos) return false;

    auto ids_ll = parse_ints(line, ids_pos.first, ids_pos.second);
    auto mask_ll = parse_ints(line, mask_pos.first, mask_pos.second);
    if (ids_ll.empty() || mask_ll.empty() || ids_ll.size() != mask_ll.size()) return false;
    ids_out.resize(ids_ll.size());
    mask_out.resize(mask_ll.size());
    for (size_t i = 0; i < ids_ll.size(); ++i) {
        ids_out[i] = static_cast<int32_t>(ids_ll[i]);
        mask_out[i] = static_cast<uint8_t>(mask_ll[i] ? 1 : 0);
    }
    return true;
}

void WikiText2Dataset::load_jsonl_split(Split split) {
    jsonl_samples_.clear();
    std::string path;
    if (split == Split::Train) path = cfg_.jsonl_train;
    else if (split == Split::Valid) path = cfg_.jsonl_valid;
    else path = cfg_.jsonl_test;

    if (path.empty()) {
        return;
    }
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Open JSONL failed: " + path);
    std::string line;
    while (std::getline(in, line)) {
        std::vector<int32_t> ids;
        std::vector<uint8_t> mask;
        if (!parse_jsonl_ids_mask(line, ids, mask)) continue;
        if (ids.size() > static_cast<size_t>(cfg_.seq_len)) {
            ids.resize(static_cast<size_t>(cfg_.seq_len));
            if (mask.size() > static_cast<size_t>(cfg_.seq_len)) mask.resize(static_cast<size_t>(cfg_.seq_len));
        }
        if (ids.empty()) continue;
        if (mask.size() != ids.size()) mask.assign(ids.size(), 0);
        JsonlSample s;
        s.ids = std::move(ids);
        s.mask = std::move(mask);
        jsonl_samples_.push_back(std::move(s));
    }
}
Batch WikiText2Dataset::next_batch(size_t batch_size, bool need_loop) {
    if (order_.empty()) {
        throw std::runtime_error("Dataset not loaded");
    }
    
    if (cursor_ >= order_.size()) {
        if (!need_loop) {
            // Return empty batch
            return Batch{nullptr, nullptr, nullptr};
        }
        shuffle(); // new epoch
    }
    
    if (jsonl_mode_) {
        const int S = cfg_.seq_len;
        if (batch_input_buffer_.size() != batch_size * S) {
            batch_input_buffer_.resize(batch_size * S);
            batch_label_buffer_.resize(batch_size * S);
            batch_attn_buffer_.resize(batch_size * S);
        }
        std::fill(batch_input_buffer_.begin(), batch_input_buffer_.end(), static_cast<int32_t>(cfg_.pad_id));
        std::fill(batch_label_buffer_.begin(), batch_label_buffer_.end(), -100);
        std::fill(batch_attn_buffer_.begin(), batch_attn_buffer_.end(), 0.0f);

        size_t start = cursor_;
        size_t end = std::min(cursor_ + batch_size, order_.size());
        size_t real_bs = end - start;
        cursor_ = end;

        for (size_t b = 0; b < real_bs; ++b) {
            const size_t ord = order_[start + b];
            const auto& sample = jsonl_samples_[ord];
            const size_t L = std::min(static_cast<size_t>(S), sample.ids.size());
            for (size_t i = 0; i < L; ++i) {
                int32_t tok = sample.ids[i];
                uint8_t m = (i < sample.mask.size()) ? sample.mask[i] : 0;
                batch_input_buffer_[b * S + i] = tok;
                batch_attn_buffer_[b * S + i] = 1.0f;
                batch_label_buffer_[b * S + i] = m ? tok : -100;
            }
        }

        Batch batch;
        batch.input_ids = from_blob(
            batch_input_buffer_.data(), 
            {static_cast<int64_t>(batch_size), static_cast<int64_t>(S)}, 
            kInt32, kCPU);
        batch.attention_mask = from_blob(
            batch_attn_buffer_.data(), 
            {static_cast<int64_t>(batch_size), static_cast<int64_t>(S)}, 
            kFloat32, kCPU);
        batch.labels = from_blob(
            batch_label_buffer_.data(), 
            {static_cast<int64_t>(batch_size), static_cast<int64_t>(S)}, 
            kInt32, kCPU);
        return batch;
    } else {
        size_t start = cursor_;
        cursor_ = std::min(cursor_ + batch_size, order_.size());
        return get_batch(start, cursor_ - start);
    }
}

std::vector<int32_t> WikiText2Dataset::peek_tokens(size_t count) const {
    if (count == 0) return {};
    
    if (cfg_.streaming_mode) {
        const_cast<WikiText2Dataset*>(this)->load_window_from_file(
            0, std::max(count + 1, static_cast<size_t>(cfg_.seq_len + 1))
        );
    }
    
    if (ids_.empty() || ids_global_offset_ > 0) {
        return {};
    }
    size_t take = std::min(count, ids_.size());
    return std::vector<int32_t>(ids_.begin(), ids_.begin() + take);
}

Batch WikiText2Dataset::get_batch(size_t index_start, size_t batch_size) const {
    const int S = cfg_.seq_len;
    
    // 🚀 Reuse buffers (reduce allocations)
    if (batch_input_buffer_.size() != batch_size * S) {
        batch_input_buffer_.resize(batch_size * S);
        batch_label_buffer_.resize(batch_size * S);
        batch_attn_buffer_.resize(batch_size * S);
    }
    
    // Zero / init
    std::fill(batch_input_buffer_.begin(), batch_input_buffer_.end(), static_cast<int32_t>(cfg_.pad_id));
    std::fill(batch_label_buffer_.begin(), batch_label_buffer_.end(), -100);
    std::fill(batch_attn_buffer_.begin(), batch_attn_buffer_.end(), 0.0f);

    for (size_t b = 0; b < batch_size; ++b) {
        const size_t ord = order_[index_start + b];
        const size_t global_start = starts_[ord];
        
        // 🚀 Streaming mode: load window on demand
        if (cfg_.streaming_mode) {
            // Ensure current chunk is inside cached window
            if (global_start < ids_global_offset_ || 
                global_start + S + 1 > ids_global_offset_ + ids_.size()) {
                // Need to reload window
                size_t window_end = std::min(global_start + cfg_.max_cache_tokens, total_tokens_);
                const_cast<WikiText2Dataset*>(this)->load_window_from_file(
                    global_start, window_end - global_start
                );
            }
        }
        
        // Compute offset inside cache
        size_t local_start = global_start - ids_global_offset_;
        
        // Actual available length (up to S+1), for label computation
        size_t avail = std::min(static_cast<size_t>(S + 1), ids_.size() - local_start);
        size_t tok_len = (avail >= 2) ? (avail - 1) : 0;
        
        // Copy inputs/labels
        for (size_t i = 0; i < tok_len && i < static_cast<size_t>(S); ++i) {
            int32_t in = ids_[local_start + i];
            // Align labels with current input token; shift is applied inside lm_cross_entropy (HF-aligned)
            int32_t lab = ids_[local_start + i];
            batch_input_buffer_[b * S + i] = in;
            batch_label_buffer_[b * S + i] = lab;
            batch_attn_buffer_[b * S + i] = 1.0f;
        }
    }

    // 🚀 Reuse tensors (wrap with from_blob to avoid copies)
    Batch batch;
    batch.input_ids = from_blob(
        batch_input_buffer_.data(), 
        {static_cast<int64_t>(batch_size), static_cast<int64_t>(S)}, 
        kInt32, kCPU);
    
    batch.attention_mask = from_blob(
        batch_attn_buffer_.data(), 
        {static_cast<int64_t>(batch_size), static_cast<int64_t>(S)}, 
        kFloat32, kCPU);
    
    batch.labels = from_blob(
        batch_label_buffer_.data(), 
        {static_cast<int64_t>(batch_size), static_cast<int64_t>(S)}, 
        kInt32, kCPU);
    
    return batch;
}

void WikiText2Dataset::ensure_pretokenized_meta_loaded() {
    if (pretokenized_meta_.loaded) return;
    if (cfg_.pretokenized_path.empty()) {
        throw std::runtime_error("pretokenized_path is empty");
    }

    std::string meta_path = resolve_meta_path(cfg_.pretokenized_path, cfg_.pretokenized_meta);
    if (meta_path.empty()) {
        throw std::runtime_error("Cannot locate meta.json for pretokenized dataset");
    }

    std::ifstream in(meta_path, std::ios::in);
    if (!in) {
        throw std::runtime_error("Open pretokenized meta failed: " + meta_path);
    }
    std::string content((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());

    pretokenized_meta_.meta_path = meta_path;
    pretokenized_meta_.loaded = true;
    pretokenized_meta_.pad_id = cfg_.pad_id;
    pretokenized_meta_.eos_id = cfg_.eos_id;
    pretokenized_meta_.insert_eos_between_lines = cfg_.insert_eos_between_lines;

    int64_t total_tokens = 0;
    if (!extract_int64_field(content, "total_tokens", total_tokens)) {
        throw std::runtime_error("meta.json missing total_tokens");
    }
    pretokenized_meta_.total_tokens = static_cast<size_t>(std::max<int64_t>(0, total_tokens));

    int64_t eos_id = -1;
    if (extract_int64_field(content, "eos_token_id", eos_id)) {
        pretokenized_meta_.eos_id = static_cast<int32_t>(eos_id);
    }
    int64_t pad_id = -1;
    if (extract_int64_field(content, "pad_token_id", pad_id)) {
        pretokenized_meta_.pad_id = static_cast<int32_t>(pad_id);
    }
    int64_t bos_id = -1;
    if (extract_int64_field(content, "bos_token_id", bos_id)) {
        pretokenized_meta_.bos_id = static_cast<int32_t>(bos_id);
    }
    int64_t unk_id = -1;
    if (extract_int64_field(content, "unk_token_id", unk_id)) {
        pretokenized_meta_.unk_id = static_cast<int32_t>(unk_id);
    }
    int64_t vocab_size = -1;
    if (extract_int64_field(content, "vocab_size", vocab_size)) {
        pretokenized_meta_.vocab_size = static_cast<int32_t>(vocab_size);
    }
    bool insert_eos_flag = true;
    if (extract_bool_field(content, "insert_eos_between_lines", insert_eos_flag)) {
        pretokenized_meta_.insert_eos_between_lines = insert_eos_flag;
    }

    size_t offset = 0;
    size_t length = 0;
    if (extract_split_range(content, "train", offset, length)) {
        pretokenized_meta_.train = {offset, length, true};
    }
    if (extract_split_range(content, "valid", offset, length) ||
        extract_split_range(content, "validation", offset, length)) {
        pretokenized_meta_.valid = {offset, length, true};
    }
    if (extract_split_range(content, "test", offset, length)) {
        pretokenized_meta_.test = {offset, length, true};
    }

    cfg_.eos_id = pretokenized_meta_.eos_id;
    cfg_.pad_id = pretokenized_meta_.pad_id;
    cfg_.insert_eos_between_lines = pretokenized_meta_.insert_eos_between_lines;
}

void WikiText2Dataset::load_pretokenized_split(Split split) {
    const PretokenizedSplit* split_info = nullptr;
    if (split == Split::Train) split_info = &pretokenized_meta_.train;
    else if (split == Split::Valid) split_info = &pretokenized_meta_.valid;
    else split_info = &pretokenized_meta_.test;

    if (!split_info || !split_info->available) {
        throw std::runtime_error("Requested split not found in pretokenized meta");
    }

    size_t use_len = split_info->length;
    float frac = std::clamp(cfg_.data_fraction, 0.0f, 1.0f);
    if (frac < 1.0f) {
        size_t min_tokens = static_cast<size_t>(cfg_.seq_len + 1);
        size_t limited = static_cast<size_t>(static_cast<double>(use_len) * frac);
        use_len = std::max(min_tokens, std::min(use_len, limited));
    }

    if (split_info->offset >= pretokenized_meta_.total_tokens) {
        throw std::runtime_error("Pretokenized split offset exceeds total tokens");
    }
    size_t max_len = pretokenized_meta_.total_tokens - split_info->offset;
    use_len = std::min(use_len, max_len);

    ids_.assign(use_len, 0);
    std::ifstream bin(cfg_.pretokenized_path, std::ios::binary);
    if (!bin) {
        throw std::runtime_error("Open pretokenized bin failed: " + cfg_.pretokenized_path);
    }

    const std::streamoff byte_offset =
        static_cast<std::streamoff>(split_info->offset * sizeof(int32_t));
    bin.seekg(byte_offset, std::ios::beg);
    bin.read(reinterpret_cast<char*>(ids_.data()),
             static_cast<std::streamsize>(use_len * sizeof(int32_t)));
    size_t read_bytes = static_cast<size_t>(bin.gcount());
    size_t read_count = read_bytes / sizeof(int32_t);
    if (read_count < use_len) {
        ids_.resize(read_count);
    }

    if (ids_.empty()) {
        throw std::runtime_error("Loaded zero tokens for pretokenized split");
    }

    ids_global_offset_ = 0;
    total_tokens_ = ids_.size();
    build_chunk_indices(ids_);
    std::cout << "[Pretokenized] split=" << (split == Split::Train ? "train" :
                    (split == Split::Valid ? "valid" : "test"))
              << " tokens=" << total_tokens_
              << " chunks=" << starts_.size() << std::endl;
}

void WikiText2Dataset::precompute_chunk_offsets() {
    // 🚀 Pre-scan: tokenize line by line, only accumulate offsets (temp allocations released immediately)
    std::ifstream file(current_file_path_);
    if (!file) throw std::runtime_error("Cannot open file: " + current_file_path_);
    
    size_t token_count = 0;
    std::string line;
    
    // Tokenize line by line, accumulate token counts only (do not store tokens)
    while (std::getline(file, line)) {
        if (!cfg_.insert_eos_between_lines && trim(line).empty()) continue;
        
        auto enc = encode_fn_(line);
        token_count += enc.size();
        
        if (cfg_.insert_eos_between_lines) {
            token_count += 1;  // EOS
        }
        
        // enc is released when leaving scope
    }
    
    // Ensure EOS at the end
    token_count += 1;
    float frac = std::clamp(cfg_.data_fraction, 0.0f, 1.0f);
    size_t limit = static_cast<size_t>(static_cast<double>(token_count) * frac);
    size_t min_tokens = static_cast<size_t>(cfg_.seq_len + 1);
    limit = std::max(limit, min_tokens);
    limit = std::min(limit, token_count);
    total_tokens_ = limit;
    
    // Generate chunk start offsets (true offsets)
    starts_.clear();
    size_t S = cfg_.seq_len;
    size_t stride = (cfg_.stride <= 0) ? S : cfg_.stride;
    
    for (size_t i = 0; i + S + 1 <= total_tokens_; i += stride) {
        starts_.push_back(i);
    }
    
    // Pre-scan done; file closed, no resident memory
}

void WikiText2Dataset::load_window_from_file(size_t global_token_start, size_t num_tokens) {
    // 🚀 Load window on demand from file (do not keep full content)
    std::ifstream file(current_file_path_);
    if (!file) throw std::runtime_error("Cannot open file: " + current_file_path_);

    size_t token_limit = total_tokens_;
    if (global_token_start >= token_limit) {
        ids_.clear();
        ids_global_offset_ = global_token_start;
        return;
    }

    size_t window_end_global = std::min(global_token_start + num_tokens, token_limit);
    size_t needed_tokens = window_end_global - global_token_start;
    if (needed_tokens == 0) {
        ids_.clear();
        ids_global_offset_ = global_token_start;
        return;
    }
    
    // Tokenize line by line, skipping unused parts
    std::vector<int32_t> window_tokens;
    window_tokens.reserve(needed_tokens + 1000);  // leave headroom
    
    size_t current_offset = 0;
    std::string line;
    
    while (std::getline(file, line)) {
        if (!cfg_.insert_eos_between_lines && trim(line).empty()) continue;
        
        auto enc = encode_fn_(line);
        if (cfg_.insert_eos_between_lines) {
            enc.push_back(cfg_.eos_id);
        }
        
        // Check if this line overlaps the requested window
        size_t line_end = current_offset + enc.size();
        
        if (line_end > global_token_start) {
            // This line is partially or fully inside the window
            size_t copy_start = (current_offset >= global_token_start) ? 
                                 0 : (global_token_start - current_offset);
            size_t max_copy = (line_end <= window_end_global) ? enc.size()
                                : copy_start + (window_end_global - std::max(current_offset, global_token_start));
            size_t copy_end = std::min(enc.size(), max_copy);
            
            for (size_t i = copy_start; i < copy_end; ++i) {
                window_tokens.push_back(enc[i]);
            }
            
            // Stop early if we have collected enough
            if (window_tokens.size() >= needed_tokens) break;
        }
        
        current_offset = line_end;
        if (current_offset >= token_limit) break;
        
        // Skip ahead until the window is reached
        if (current_offset < global_token_start) continue;
    }
    
    // Shrink temporary window capacity before moving it
    window_tokens.shrink_to_fit();
    // Update cache
    if (window_tokens.size() > needed_tokens) {
        window_tokens.resize(needed_tokens);
    }
    ids_ = std::move(window_tokens);
    ids_global_offset_ = global_token_start;
    
    // Cap size
    if (ids_.size() > cfg_.max_cache_tokens) {
        ids_.resize(cfg_.max_cache_tokens);
    }
    // Release extra capacity to avoid RSS bloat
    ids_.shrink_to_fit();
    
    // Monitor window size and offset (helps detect oversized/overlapping windows)
    static size_t __wt2_load_count = 0;
    if (((++__wt2_load_count) % 100) == 1) {
        std::cout << "[WT2] window loaded: tokens=" << ids_.size()
                  << " global_offset=" << ids_global_offset_ << std::endl;
    }
}

}  // namespace ops
