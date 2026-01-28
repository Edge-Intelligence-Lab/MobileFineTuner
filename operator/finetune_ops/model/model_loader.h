#pragma once

#include "../core/tensor.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>

namespace ops {

struct ModelConfig {
    std::string model_type;
    std::string model_name;
    std::string vocab_file;

    ModelConfig() = default;
    virtual ~ModelConfig() = default;

    virtual void save(const std::string& path) const = 0;
    virtual void load(const std::string& path) = 0;
};

class ModelLoader {
public:
    ModelLoader() = default;
    virtual ~ModelLoader() = default;

    virtual std::unique_ptr<ModelConfig> load_config(
        const std::string& path) = 0;

    virtual std::unordered_map<std::string, TensorPtr> load_weights(
        const std::string& path) = 0;

    virtual void save_config(
        const ModelConfig& config,
        const std::string& path) = 0;

    virtual void save_weights(
        const std::unordered_map<std::string, TensorPtr>& weights,
        const std::string& path) = 0;

    virtual std::vector<std::string> load_vocab(
        const std::string& path) = 0;

    virtual void save_vocab(
        const std::vector<std::string>& vocab,
        const std::string& path) = 0;

protected:

    virtual void validate_config(const ModelConfig& config) = 0;
    virtual void validate_weights(
        const std::unordered_map<std::string, TensorPtr>& weights,
        const ModelConfig& config) = 0;
};

class GGUFModelLoader : public ModelLoader {
public:
    GGUFModelLoader() = default;
    ~GGUFModelLoader() override = default;

    std::unique_ptr<ModelConfig> load_config(
        const std::string& path) override;

    std::unordered_map<std::string, TensorPtr> load_weights(
        const std::string& path) override;

    void save_config(
        const ModelConfig& config,
        const std::string& path) override;

    void save_weights(
        const std::unordered_map<std::string, TensorPtr>& weights,
        const std::string& path) override;

    std::vector<std::string> load_vocab(
        const std::string& path) override;

    void save_vocab(
        const std::vector<std::string>& vocab,
        const std::string& path) override;

protected:
    void validate_config(const ModelConfig& config) override;
    void validate_weights(
        const std::unordered_map<std::string, TensorPtr>& weights,
        const ModelConfig& config) override;

private:

    void read_gguf_header(std::ifstream& file);
    void write_gguf_header(std::ofstream& file);
    void read_tensor_data(std::ifstream& file, TensorPtr tensor);
    void write_tensor_data(std::ofstream& file, const TensorPtr& tensor);
};

class SafeTensorsModelLoader : public ModelLoader {
public:
    SafeTensorsModelLoader() = default;
    ~SafeTensorsModelLoader() override = default;

    std::unique_ptr<ModelConfig> load_config(
        const std::string& path) override;

    std::unordered_map<std::string, TensorPtr> load_weights(
        const std::string& path) override;

    void save_config(
        const ModelConfig& config,
        const std::string& path) override;

    void save_weights(
        const std::unordered_map<std::string, TensorPtr>& weights,
        const std::string& path) override;

    std::vector<std::string> load_vocab(
        const std::string& path) override;

    void save_vocab(
        const std::vector<std::string>& vocab,
        const std::string& path) override;

protected:
    void validate_config(const ModelConfig& config) override;
    void validate_weights(
        const std::unordered_map<std::string, TensorPtr>& weights,
        const ModelConfig& config) override;

private:

    void parse_header(const std::string& header_json);
    void write_header(std::ofstream& file,
                     const std::unordered_map<std::string, TensorPtr>& weights);
};

}
