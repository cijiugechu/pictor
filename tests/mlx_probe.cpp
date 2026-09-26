#include "mlx/text_encoder.hpp"
#include "mlx/transformer.hpp"
#include "mlx/vae.hpp"
#include "stable-diffusion.h"
#include <fstream>
#include <iostream>

// Model-dependent numerical probe; deliberately outside the no-exception API.
int main(int argc, char **argv) {
    using namespace pictor::mlx_backend;
    try {
        if (argc != 3)
            throw std::invalid_argument("usage: mlx_probe MODEL_DIR FIXTURE_DIR");
        mx::set_default_device(mx::Device(mx::Device::gpu));
        sd_set_log_callback([](sd_log_level_t, const char *, void *) {}, nullptr);
        const std::filesystem::path model = argv[1], fixture = argv[2];
        auto input = mx::load_safetensors((fixture / "inputs.safetensors").string()).first;
        TextEncoder text(model / "text_encoder");
        const std::string prompt =
            "A small red fox sitting on a mossy rock in a sunlit forest, detailed fur, soft natural light";
        const auto embed = text.encode(prompt);
        mx::eval(embed);
        mx::save_safetensors((fixture / "cpp-text.safetensors").string(), {{"text", embed}});
        std::ofstream tokens(fixture / "cpp-tokens.txt");
        for (const auto &p : {prompt, std::string("一只红狐狸，在雪地中。 café\n✨"),
                              std::string("  (fox:1.2)  don't  12345"), std::string("<|im_start|>fox")}) {
            for (int t : text.tokens(p))
                tokens << t << ' ';
            tokens << '\n';
        }
        Transformer dit(model / "transformer");
        const auto noise = dit.predict(input.at("latents"), input.at("text"), input.at("timestep"), input.at("ids"));
        mx::eval(noise);
        mx::save_safetensors((fixture / "cpp-dit.safetensors").string(), {{"noise", noise}});
        Vae vae(model / "vae");
        const auto decoded = vae.decode(input.at("packed"), false);
        const auto encoded = vae.encode(input.at("pixels"));
        mx::eval({decoded, encoded});
        mx::save_safetensors((fixture / "cpp-vae.safetensors").string(), {{"decoded", decoded}, {"encoded", encoded}});
        Vae small("models/flux2-klein-4b/full_encoder_small_decoder.safetensors");
        mx::save_safetensors(
            (fixture / "cpp-small.safetensors").string(),
            {{"decoded", small.decode(input.at("packed"), false)}, {"encoded", small.encode(input.at("pixels"))}});
        std::cout << "MLX component probe completed\n";
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
