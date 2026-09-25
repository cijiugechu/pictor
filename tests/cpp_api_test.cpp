#include <pictor/anima.hpp>
#include <cstdio>
#include <cstdlib>

#define CHECK(expression) do { if (!(expression)) { std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #expression); return 1; } } while (0)

int main() {
    using namespace pictor;
    std::unique_ptr<AnimaSession> session;
    auto status = AnimaSession::create({"/nonexistent-pictor-cpp-model.gguf"}, session);
    CHECK(status.code == ErrorCode::invalid_argument && !session);
    auto request = preset_request(Preset::fast);
    CHECK(!validate_request(request));
    request.prompt = "cat";
    CHECK(validate_request(request));
    Image image;
    CHECK(write_png("unused.png", image).code == ErrorCode::invalid_argument);
    image = {1, 1, 3, {1, 2, 3}, 0, 0};
    CHECK(write_png("/dev/null/pictor.png", image).code == ErrorCode::io_error);
    CHECK(write_png("", image).code == ErrorCode::invalid_argument);
    status = failure(ErrorCode::backend_error, std::string(1024, 'a'));
    CHECK(std::strlen(status.message) == 511 && status.message[511] == '\0');
    std::puts("PASS: C++ explicit status API with exceptions disabled");
}
