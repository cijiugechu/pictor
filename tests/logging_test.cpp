#include "logging.hpp"

int main() {
    // Messages must remain data, including braces; named loggers must use stderr.
    pictor::logging::app().info("{}", "literal {braces}");
    pictor::logging::backend().warn("backend warning");
    pictor::logging::backend().debug("backend detail");
}
