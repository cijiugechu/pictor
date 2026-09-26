// Independent pinned ggml stage oracle for the dequantized P3 MLX port.
#include "anima.hpp"
#include "ggml-metal.h"
#include "llm.hpp"
#include "model.h"
#include "wan.hpp"
#include <filesystem>
#include <mlx/mlx.h>
#include <stdexcept>
namespace mx = mlx::core;
static void require(bool ok) {
    if (!ok)
        throw std::runtime_error("ggml stage probe failed");
}
static sd::Tensor<float> tensor(const mx::array &x, std::vector<int64_t> shape) {
    auto a = mx::astype(x, mx::float32);
    mx::eval(a);
    return sd::Tensor<float>(shape, std::vector<float>(a.data<float>(), a.data<float>() + a.size()));
}
int main(int argc, char **argv) try {
    if (argc != 3)
        return 2;
    std::filesystem::path p = argv[2];
    auto inputs = mx::load_safetensors((p / "inputs.safetensors").string()).first;
    ModelLoader loader;
    require(loader.init_from_file_and_convert_name(argv[1]));
    auto backend = ggml_backend_metal_init();
    require(backend != nullptr);
    std::unordered_map<std::string, mx::array> output;
    {
        LLM::LLMRunner net(LLM::LLMArch::QWEN3, backend, false, loader.get_tensor_storage_map(), "text_encoders.llm",
                           false);
        std::map<std::string, ggml_tensor *> tensors;
        net.get_param_tensors(tensors, "text_encoders.llm");
        require(net.alloc_params_buffer());
        require(loader.load_tensors(tensors, {""}, 4));
        net.set_flash_attention_enabled(true);
        auto ids = inputs.at("qwen_ids");
        mx::eval(ids);
        sd::Tensor<int32_t> tid({int64_t(ids.size()), 1},
                                std::vector<int32_t>(ids.data<int>(), ids.data<int>() + ids.size()));
        auto y = net.compute(4, tid, {}, {}, {});
        require(!y.empty());
        output.emplace("text", mx::array(y.data(), {1, int(ids.size()), 1024}));
        mx::eval(output.at("text"));
    }
    {
        Anima::AnimaRunner net(backend, false, loader.get_tensor_storage_map());
        std::map<std::string, ggml_tensor *> tensors;
        net.get_param_tensors(tensors, "model.diffusion_model");
        require(net.alloc_params_buffer());
        require(loader.load_tensors(tensors, {""}, 4));
        net.set_flash_attention_enabled(true);
        auto y = net.compute(4, tensor(inputs.at("latent"), {8, 8, 16, 1}), sd::Tensor<float>({1}, {0.5f}),
                             tensor(inputs.at("context"), {1024, 512, 1}));
        require(!y.empty());
        output.emplace("noise", mx::array(y.data(), {1, 16, 8, 8}));
        mx::eval(output.at("noise"));
    }
    {
        WAN::WanVAERunner net(backend, false, loader.get_tensor_storage_map(), "first_stage_model", true,
                              VERSION_ANIMA);
        std::map<std::string, ggml_tensor *> tensors;
        net.get_param_tensors(tensors, "first_stage_model");
        require(net.alloc_params_buffer());
        require(loader.load_tensors(tensors, {""}, 4));
        net.set_flash_attention_enabled(true);
        auto z = net.diffusion_to_vae_latents(tensor(inputs.at("latent"), {8, 8, 16, 1}));
        sd_tiling_params_t tiling{};
        auto y = net.decode(4, z, tiling);
        require(!y.empty());
        output.emplace("decoded", mx::transpose(mx::array(y.data(), {1, 3, 64, 64}), {0, 2, 3, 1}));
        mx::eval(output.at("decoded"));
    }
    mx::save_safetensors((p / "ggml.safetensors").string(), output);
    ggml_backend_free(backend);
    return 0;
} catch (const std::exception &e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
}
