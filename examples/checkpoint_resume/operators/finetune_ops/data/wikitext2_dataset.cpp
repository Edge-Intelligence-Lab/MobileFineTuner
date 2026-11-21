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
 #include <cmath>
 #include <limits>
 #include <locale>
 #include <set>
 #include "../../third_party/nlohmann/json.hpp"
 
 namespace ops {
 
namespace {

using StateJson = nlohmann::json;

void resume_require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(std::string("WikiText resume: ") + message);
}

StateJson parse_resume_json(const std::string& text) {
    std::vector<std::set<std::string>> keys;
    return StateJson::parse(text, [&](int, StateJson::parse_event_t event, StateJson& value) {
        if (event == StateJson::parse_event_t::object_start) keys.emplace_back();
        if (event == StateJson::parse_event_t::key)
            resume_require(keys.back().insert(value.get<std::string>()).second, "duplicate JSON key");
        if (event == StateJson::parse_event_t::object_end) keys.pop_back();
        return true;
    });
}

size_t resume_size(const StateJson& value) {
    resume_require(value.is_number_unsigned(), "expected unsigned integer");
    auto n = value.get<uint64_t>();
    resume_require(n <= std::numeric_limits<size_t>::max(), "integer overflow");
    return static_cast<size_t>(n);
}

std::string rng_state(const std::mt19937_64& rng) {
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << rng;
    return out.str();
}

std::mt19937_64 restore_rng(const StateJson& value) {
    resume_require(value.is_string(), "expected RNG string");
    const auto text = value.get<std::string>();
    std::istringstream words(text);
    words.imbue(std::locale::classic());
    uint64_t word = 0;
    size_t count = 0;
    bool nonzero = false;
    while (words >> word) {
        if (count < std::mt19937_64::state_size) nonzero = nonzero || word != 0;
        else resume_require(count == std::mt19937_64::state_size && word <= std::mt19937_64::state_size,
                            "RNG position out of range");
        ++count;
    }
    resume_require(nonzero && (count == std::mt19937_64::state_size || count == std::mt19937_64::state_size + 1),
                   "invalid RNG word count/state");
    std::istringstream in(text);
    in.imbue(std::locale::classic());
    std::mt19937_64 rng;
    resume_require(static_cast<bool>(in >> rng), "invalid RNG state");
    resume_require(rng_state(rng) == text, "noncanonical/trailing RNG state");
    return rng;
}
 
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

bool extract_array_field(const std::string& text,
                         const std::string& key,
                         std::vector<int32_t>& out) {
    out.clear();
    auto key_pos = text.find("\"" + key + "\"");
    if (key_pos == std::string::npos) return false;
    auto colon = text.find(':', key_pos);
    if (colon == std::string::npos) return false;
    auto lb = text.find('[', colon);
    if (lb == std::string::npos) return false;
    auto rb = text.find(']', lb);
    if (rb == std::string::npos) return false;
    std::string body = text.substr(lb + 1, rb - lb - 1);
    std::stringstream ss(body);
    std::string item;
    while (std::getline(ss, item, ',')) {
        size_t l = 0, r = item.size();
        while (l < r && std::isspace(static_cast<unsigned char>(item[l]))) ++l;
        while (r > l && std::isspace(static_cast<unsigned char>(item[r - 1]))) --r;
        auto v = item.substr(l, r - l);
        if (v.empty()) continue;
        try {
            out.push_back(static_cast<int32_t>(std::stoll(v)));
        } catch (...) {
            return false;
        }
    }
    return true;
}

}  // namespace
 
 static std::string trim(const std::string& s) {
     size_t l = 0, r = s.size();
     while (l < r && std::isspace(static_cast<unsigned char>(s[l]))) ++l;
     while (r > l && std::isspace(static_cast<unsigned char>(s[r-1]))) --r;
     return s.substr(l, r-l);
 }
 
 WikiText2Dataset::WikiText2Dataset(const WT2Config& cfg, HFTokenizer* tok)
     : cfg_(cfg), cursor_(0), rng_(cfg.seed) {
     encode_fn_ = [tok](const std::string& text) {
         return tok->encode(text);
     };
 }
 
 WikiText2Dataset::WikiText2Dataset(const WT2Config& cfg,
                                    std::function<std::vector<int32_t>(const std::string&)> encode_fn)
     : cfg_(cfg), encode_fn_(std::move(encode_fn)),
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
        // Raw version allows empty lines; whether to skip is controlled by insert_eos_between_lines
        if (!cfg_.insert_eos_between_lines && trim(line).empty()) continue;
        lines.emplace_back(line);
    }
     return lines;
 }
 
std::vector<int32_t> WikiText2Dataset::tokenize_and_pack(const std::vector<std::string>& lines) const {
    std::vector<int32_t> ids;
    ids.reserve(lines.size() * 16); // Rough estimate

    for (size_t i = 0; i < lines.size(); ++i) {
        // Use GPT2BPETokenizer (returns std::vector<int>)
        auto enc = encode_fn_(lines[i]);
        ids.insert(ids.end(), enc.begin(), enc.end());
        
        if (cfg_.insert_eos_between_lines) {
            ids.push_back(static_cast<int32_t>(cfg_.eos_id));
        }
    }
    
    // Ensure one more token at the end for labels right shift
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

    // Need S+1 tokens to form [S] input + [S] labels (right shift)
    const long long need = static_cast<long long>(S) + 1;
    long long N = static_cast<long long>(ids.size());

    for (long long s = 0; s + need <= N; s += stride) {
        starts_.push_back(static_cast<size_t>(s));
    }

    if (!cfg_.drop_last) {
        // Add one tail chunk (may be incomplete), PAD in get_batch
        if (starts_.empty() || starts_.back() + need < static_cast<size_t>(N)) {
            long long s = std::max(0LL, N - need);
            if (starts_.empty() || starts_.back() != static_cast<size_t>(s)) {
                starts_.push_back(static_cast<size_t>(s));
            }
        }
    }

    // Construct traversal order (can be shuffled)
     order_.resize(starts_.size());
     std::iota(order_.begin(), order_.end(), 0);
 }
 
 void WikiText2Dataset::load(Split split) {
     loaded_ = false;
     loaded_split_ = split;
     ids_.clear();
     starts_.clear();
     order_.clear();
     jsonl_ids_.clear();
     jsonl_mask_.clear();
     cursor_ = 0;
     ids_global_offset_ = 0;
     total_tokens_ = 0;
     pretokenized_mode_ = !cfg_.pretokenized_path.empty();
     jsonl_mode_ = false;

     if (pretokenized_mode_) {
         cfg_.streaming_mode = false;  // 预分词模式直接驻留所需切片
         ensure_pretokenized_meta_loaded();
         load_pretokenized_split(split);
         if (split == Split::Train && cfg_.shuffle_train) {
             shuffle();
         }
        loaded_ = true;
        return;
    }

    const bool has_jsonl =
        !cfg_.jsonl_train.empty() || !cfg_.jsonl_valid.empty() || !cfg_.jsonl_test.empty();
    if (has_jsonl) {
        load_jsonl_split(split);
        if (split == Split::Train && cfg_.shuffle_train) {
            shuffle();
        }
        loaded_ = true;
        return;
    }
    
    // Save file path
    if (split == Split::Train) current_file_path_ = cfg_.train_path;
    else if (split == Split::Valid) current_file_path_ = cfg_.valid_path;
    else current_file_path_ = cfg_.test_path;
    
    if (cfg_.streaming_mode) {
        // True streaming mode: prescan to compute true token offsets, no text residency
        precompute_chunk_offsets();
        
        // Construct traversal order
         order_.resize(starts_.size());
         std::iota(order_.begin(), order_.end(), 0);
         
         if (split == Split::Train && cfg_.shuffle_train) {
             shuffle();
        }
        
        // Preload first window (covers first few chunks)
        if (!starts_.empty()) {
            size_t first_window_end = std::min(
                starts_[0] + cfg_.max_cache_tokens, 
                total_tokens_
            );
            load_window_from_file(starts_[0], first_window_end - starts_[0]);
        }
    } else {
        // Original mode: load full data at once
        auto lines = read_lines_for_split(split);
         ids_ = tokenize_and_pack(lines);
         build_chunk_indices(ids_);
         
         if (split == Split::Train && cfg_.shuffle_train) {
             shuffle();
          }
      }
     loaded_ = true;
 }

std::string WikiText2Dataset::export_state() const {
    resume_require(loaded_, "load a split before checkpointing");
    resume_require(cfg_.seq_len > 0 && std::isfinite(cfg_.data_fraction) &&
                   (!cfg_.streaming_mode || cfg_.max_cache_tokens >= static_cast<size_t>(cfg_.seq_len) + 1),
                   "unsupported effective configuration");
    resume_require(cursor_ <= order_.size(), "invalid cursor");
    size_t tokens = cfg_.streaming_mode ? total_tokens_ : ids_.size();
    if (jsonl_mode_) {
        tokens = 0;
        for (const auto& ids : jsonl_ids_) tokens += ids.size();
    }
    StateJson config = {{"seq_len", cfg_.seq_len},
        {"stride", cfg_.stride <= 0 ? cfg_.seq_len : cfg_.stride},
        {"eos_id", cfg_.eos_id}, {"pad_id", cfg_.pad_id},
        {"insert_eos_between_lines", cfg_.insert_eos_between_lines},
        {"drop_last", cfg_.drop_last}, {"seed", cfg_.seed},
        {"shuffle_train", cfg_.shuffle_train}, {"streaming_mode", cfg_.streaming_mode},
        {"max_cache_tokens", cfg_.max_cache_tokens}, {"data_fraction", cfg_.data_fraction}};
    StateJson state = {{"schema", "mobilefinetuner.dataset"}, {"version", 1},
        {"dataset", "wikitext2"}, {"config", config},
        {"split", loaded_split_ == Split::Train ? "train" : loaded_split_ == Split::Valid ? "valid" : "test"},
        {"mode", jsonl_mode_ ? "jsonl" : pretokenized_mode_ ? "pretokenized" : cfg_.streaming_mode ? "streaming_text" : "text"},
        {"sequence_count", num_sequences()}, {"token_count", tokens}, {"starts", starts_},
        {"order", order_}, {"cursor", cursor_}, {"cursor_unit", "sequence"},
        {"rng_engine", "std::mt19937_64"}, {"rng_format", "classic_stream_v1"}, {"rng", rng_state(rng_)}};
    if (pretokenized_mode_) {
        const auto& part = loaded_split_ == Split::Train ? pretokenized_meta_.train :
                           loaded_split_ == Split::Valid ? pretokenized_meta_.valid : pretokenized_meta_.test;
        state["pretokenized_range"] = {{"offset", part.offset}, {"length", part.length},
                                       {"total_tokens", pretokenized_meta_.total_tokens}};
    }
    return state.dump();
}

void WikiText2Dataset::import_state(const std::string& text) {
    prepare_import(text)();
}

std::function<void()> WikiText2Dataset::prepare_import(const std::string& text) {
    auto state = parse_resume_json(text);
    auto expected = StateJson::parse(export_state());
    resume_require(state.is_object(), "expected object");
    const auto& saved_order = state.at("order");
    resume_require(saved_order.is_array() && saved_order.size() == order_.size(), "order length mismatch");
    std::vector<size_t> order;
    order.reserve(order_.size());
    std::vector<bool> seen(order_.size(), false);
    for (const auto& value : saved_order) {
        size_t n = resume_size(value);
        resume_require(n < seen.size() && !seen[n], "order is not a permutation");
        seen[n] = true;
        order.push_back(n);
    }
    size_t cursor = resume_size(state.at("cursor"));
    resume_require(cursor <= order.size(), "cursor out of range");
    auto rng = restore_rng(state.at("rng"));
    for (const char* field : {"order", "cursor", "rng"}) {
        state.erase(field);
        expected.erase(field);
    }
    // Canonical serialization also rejects float-for-integer and bool-for-number
    // coercions, as well as missing or unknown schema/config fields.
    resume_require(state.dump() == expected.dump(), "schema/config/split/count/starts mismatch");
    return [this, order = std::move(order), cursor, rng]() mutable {
        order_.swap(order);
        cursor_ = cursor;
        rng_ = rng;
        if (cfg_.streaming_mode) {
            ids_.clear();
            ids_global_offset_ = 0;
        }
    };
    // Resident ids_ are the dataset, not a cache. Keep the reusable batch buffers
    // alive as well: returned from_blob tensors may still reference their memory.
}
 
 size_t WikiText2Dataset::num_sequences() const {
     if (jsonl_mode_) return order_.size();
     return starts_.size();
 }
 
 void WikiText2Dataset::shuffle() {
     std::shuffle(order_.begin(), order_.end(), rng_);
     cursor_ = 0;
 }
 
 void WikiText2Dataset::reset_cursor() {
     cursor_ = 0;
 }
 
Batch WikiText2Dataset::next_batch(size_t batch_size, bool need_loop) {
     if (order_.empty()) {
         throw std::runtime_error("Dataset not loaded");
     }

    auto reset_for_loop = [&]() {
        if (cfg_.shuffle_train) {
            shuffle();
        } else {
            cursor_ = 0;
        }
    };

    if (cursor_ >= order_.size()) {
        if (!need_loop) {
            return Batch{nullptr, nullptr, nullptr};
        }
        reset_for_loop();
    }

    if (cfg_.drop_last && cursor_ + batch_size > order_.size()) {
        if (!need_loop || order_.size() < batch_size) {
            return Batch{nullptr, nullptr, nullptr};
        }
        reset_for_loop();
        if (order_.size() < batch_size) {
            return Batch{nullptr, nullptr, nullptr};
        }
    }

    size_t start = cursor_;
    cursor_ = std::min(cursor_ + batch_size, order_.size());
    return get_batch(start, cursor_ - start);
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
    
    // Reuse buffers (reduce allocations)
    if (batch_input_buffer_.size() != batch_size * S) {
        batch_input_buffer_.resize(batch_size * S);
        batch_label_buffer_.resize(batch_size * S);
        batch_attn_buffer_.resize(batch_size * S);
    }
    
    // Zero/initialize
    std::fill(batch_input_buffer_.begin(), batch_input_buffer_.end(), static_cast<int32_t>(cfg_.pad_id));
    std::fill(batch_label_buffer_.begin(), batch_label_buffer_.end(), -100);
    std::fill(batch_attn_buffer_.begin(), batch_attn_buffer_.end(), 0.0f);
 
    if (jsonl_mode_) {
        for (size_t b = 0; b < batch_size; ++b) {
            const size_t ord = order_[index_start + b];
            const auto& ids = jsonl_ids_[ord];
            const auto& mask = jsonl_mask_[ord];
            for (size_t i = ids.size(); i < static_cast<size_t>(S); ++i) {
                batch_input_buffer_[b * S + i] = static_cast<int32_t>(cfg_.pad_id);
                batch_label_buffer_[b * S + i] = -100;
                batch_attn_buffer_[b * S + i] = 0.0f;
            }
            for (size_t i = 0; i < ids.size(); ++i) {
                batch_input_buffer_[b * S + i] = ids[i];
                batch_attn_buffer_[b * S + i] = 1.0f;
                batch_label_buffer_[b * S + i] = (mask[i] > 0) ? ids[i] : -100;
            }
        }
    } else {
    for (size_t b = 0; b < batch_size; ++b) {
        const size_t ord = order_[index_start + b];
        const size_t global_start = starts_[ord];
        
        // Streaming mode: load window on demand
        if (cfg_.streaming_mode) {
            // Ensure current chunk is in cache window
            if (global_start < ids_global_offset_ || 
                global_start + S + 1 > ids_global_offset_ + ids_.size()) {
                // Need to reload window
                size_t window_end = std::min(global_start + cfg_.max_cache_tokens, total_tokens_);
                const_cast<WikiText2Dataset*>(this)->load_window_from_file(
                    global_start, window_end - global_start
                );
            }
        }
        
        // Calculate offset in cache
        size_t local_start = global_start - ids_global_offset_;
        
        // True available length (at most S+1), for calculating labels
        size_t avail = std::min(static_cast<size_t>(S + 1), ids_.size() - local_start);
        size_t tok_len = (avail >= 2) ? (avail - 1) : 0;
        
        // Copy input/labels
        for (size_t i = 0; i < tok_len && i < static_cast<size_t>(S); ++i) {
            int32_t in = ids_[local_start + i];
            // Adjusted to "no pre-shift": label aligns with current input token
            // Actual right shift is done inside lm_cross_entropy (aligns with HF behavior)
            int32_t lab = ids_[local_start + i];
             batch_input_buffer_[b * S + i] = in;
             batch_label_buffer_[b * S + i] = lab;
             batch_attn_buffer_[b * S + i] = 1.0f;
        }
    }
    }

    // Reuse Tensor (use from_blob wrapper, avoid copying each time)
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

void WikiText2Dataset::load_jsonl_split(Split split) {
    cfg_.streaming_mode = false;
    jsonl_mode_ = true;

    if (split == Split::Train) current_file_path_ = cfg_.jsonl_train;
    else if (split == Split::Valid) current_file_path_ = cfg_.jsonl_valid;
    else current_file_path_ = cfg_.jsonl_test;

    if (current_file_path_.empty()) {
        throw std::runtime_error("JSONL path missing for requested split");
    }

    std::ifstream in(current_file_path_);
    if (!in) throw std::runtime_error("Open JSONL file failed: " + current_file_path_);

    order_.clear();
    std::string line;
    size_t line_number = 0;
    while (std::getline(in, line)) {
        ++line_number;
        std::vector<int32_t> ids;
        std::vector<int32_t> mask;
        if (cfg_.jsonl_vocab_limit > 0) {
            try {
                const auto row = StateJson::parse(line);
                const auto& tokens = row.at("ids");
                const auto& weights = row.at("mask");
                if (!tokens.is_array() || !weights.is_array() || tokens.size() < 2 || tokens.size() != weights.size())
                    throw std::runtime_error("expected equal ids/mask arrays of length >= 2");
                bool target = false;
                for (size_t i = 0; i < tokens.size(); ++i) {
                    if (!tokens[i].is_number_integer() || tokens[i] < 0 || tokens[i] >= cfg_.jsonl_vocab_limit)
                        throw std::runtime_error("token outside model vocabulary");
                    if (!weights[i].is_number_integer() || (weights[i] != 0 && weights[i] != 1))
                        throw std::runtime_error("mask must be binary integers");
                    if (i < static_cast<size_t>(cfg_.seq_len)) {
                        ids.push_back(tokens[i].get<int32_t>());
                        mask.push_back(weights[i].get<int32_t>());
                        target |= i > 0 && mask.back() == 1;
                    }
                }
                if (!target) throw std::runtime_error("no supervised target after truncation");
            } catch (const std::exception& e) {
                throw std::runtime_error(current_file_path_ + ":" + std::to_string(line_number) + ": " + e.what());
            }
        } else if (!extract_array_field(line, "ids", ids) || !extract_array_field(line, "mask", mask)) {
            continue;
        }
        if (ids.size() != mask.size() || ids.empty()) continue;
        if (ids.size() > static_cast<size_t>(cfg_.seq_len)) {
            ids.resize(static_cast<size_t>(cfg_.seq_len));
            mask.resize(static_cast<size_t>(cfg_.seq_len));
        }
        jsonl_ids_.push_back(std::move(ids));
        jsonl_mask_.push_back(std::move(mask));
        order_.push_back(order_.size());
    }
    if (cfg_.jsonl_vocab_limit > 0 && (in.bad() || order_.empty()))
        throw std::runtime_error("Unreadable or empty JSONL split: " + current_file_path_);
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
    // Prescan: tokenize line by line, only accumulate offset info (temporary allocation released immediately)
    std::ifstream file(current_file_path_);
    if (!file) throw std::runtime_error("Cannot open file: " + current_file_path_);
    
    size_t token_count = 0;
    std::string line;
    
    // Tokenize line by line, only accumulate token count (don't save tokens themselves)
    while (std::getline(file, line)) {
        if (!cfg_.insert_eos_between_lines && trim(line).empty()) continue;
        
        auto enc = encode_fn_(line);
        token_count += enc.size();
        
        if (cfg_.insert_eos_between_lines) {
            token_count += 1;  // EOS
        }
        
        // enc released immediately when leaving scope
    }
    
    // Ensure EOS at the end
    token_count += 1;
     float frac = std::clamp(cfg_.data_fraction, 0.0f, 1.0f);
     size_t limit = static_cast<size_t>(static_cast<double>(token_count) * frac);
     size_t min_tokens = static_cast<size_t>(cfg_.seq_len + 1);
     limit = std::max(limit, min_tokens);
     limit = std::min(limit, token_count);
    total_tokens_ = limit;
    
    // Generate chunk start points (true offsets)
    starts_.clear();
    size_t S = cfg_.seq_len;
    size_t stride = (cfg_.stride <= 0) ? S : cfg_.stride;
    
    for (size_t i = 0; i + S + 1 <= total_tokens_; i += stride) {
        starts_.push_back(i);
    }
    
    // Prescan complete, file closed, no memory residency
}

void WikiText2Dataset::load_window_from_file(size_t global_token_start, size_t num_tokens) {
    // Read window from file on demand (no full data residency)
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
    
    // Tokenize line by line, skip unnecessary parts
    std::vector<int32_t> window_tokens;
    window_tokens.reserve(needed_tokens + 1000);  // Leave margin
    
    size_t current_offset = 0;
     std::string line;
     
     while (std::getline(file, line)) {
         if (!cfg_.insert_eos_between_lines && trim(line).empty()) continue;
         
         auto enc = encode_fn_(line);
         if (cfg_.insert_eos_between_lines) {
             enc.push_back(cfg_.eos_id);
        }
        
        // Check if entering window range
        size_t line_end = current_offset + enc.size();
        
        if (line_end > global_token_start) {
            // This line has part or all in window
            size_t copy_start = (current_offset >= global_token_start) ? 
                                 0 : (global_token_start - current_offset);
            size_t max_copy = (line_end <= window_end_global) ? enc.size()
                                : copy_start + (window_end_global - std::max(current_offset, global_token_start));
            size_t copy_end = std::min(enc.size(), max_copy);
            
            for (size_t i = copy_start; i < copy_end; ++i) {
                window_tokens.push_back(enc[i]);
            }
            
            // If collected enough, exit early
            if (window_tokens.size() >= needed_tokens) break;
        }
        
        current_offset = line_end;
        if (current_offset >= token_limit) break;
        
        // If not reached window yet, continue skipping
        if (current_offset < global_token_start) continue;
    }
    
    // Shrink temporary window capacity before handoff, avoid huge capacity being transferred
    window_tokens.shrink_to_fit();
    // Update cache
     if (window_tokens.size() > needed_tokens) {
         window_tokens.resize(needed_tokens);
     }
    ids_ = std::move(window_tokens);
    ids_global_offset_ = global_token_start;
    
    // Limit size
    if (ids_.size() > cfg_.max_cache_tokens) {
        ids_.resize(cfg_.max_cache_tokens);
    }
    // Release excess capacity, avoid capacity accumulation causing RSS growth
    ids_.shrink_to_fit();
    
    // Monitor window size and offset (helps locate if window is too large/overlaps too much)
    static size_t __wt2_load_count = 0;
     if (((++__wt2_load_count) % 100) == 1) {
         std::cout << "[WT2] window loaded: tokens=" << ids_.size()
                   << " global_offset=" << ids_global_offset_ << std::endl;
     }
 }
 
 }  // namespace ops
 
